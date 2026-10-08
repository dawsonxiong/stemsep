"""HTDemucs, split into the parts that need different handling.

`HTDemucs.forward` mixes three kinds of work: an STFT on the way in, the
network itself, and a masked iSTFT on the way out. The STFTs work on complex
tensors, which half precision, `torch.compile` and ONNX all handle badly, and
they're cheap. So they stay in float32 PyTorch (`pre` and `post`), and only
`Core`, the network, is handed to a faster backend.

The three together compute exactly what `HTDemucs.forward` does for an input
of the model's training length, which is the only length `Separator` uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def load(name: str = "htdemucs"):
    """The pretrained model `name`, as demucs ships it: a `BagOfModels`, even
    when the bag holds one model."""
    from demucs.pretrained import get_model

    bag = get_model(name)
    bag.eval()
    for model in bag.models:
        _check_supported(model)
    return bag


def _check_supported(model) -> None:
    from demucs.htdemucs import HTDemucs

    if not isinstance(model, HTDemucs):
        raise TypeError(f"stemsep only runs HTDemucs models, not {type(model).__name__}")
    if not model.cac or not model.use_train_segment:
        raise TypeError("stemsep needs a model with cac=True and use_train_segment=True")


def segment_length(model) -> int:
    """Samples per chunk: the length the model was trained on."""
    return int(model.segment * model.samplerate)


@dataclass
class Spec:
    """What `pre` hands to `Core`, and what `post` needs to undo it."""

    z: torch.Tensor  # mixture spectrogram, complex [B, C, Fr, T]
    x: torch.Tensor  # normalized spectrogram as channels, [B, 2C, Fr, T]
    xt: torch.Tensor  # normalized waveform, [B, C, L]
    mean: torch.Tensor
    std: torch.Tensor
    meant: torch.Tensor
    stdt: torch.Tensor


def pre(model, mix: torch.Tensor) -> Spec:
    """STFT and normalization, from the top of `HTDemucs.forward`."""
    z = model._spec(mix)
    # On MPS, demucs's STFT runs on the CPU (no complex support), so `z` is
    # there; the network's input goes back to the mix's device.
    x = model._magnitude(z).to(mix.device)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    meant = mix.mean(dim=(1, 2), keepdim=True)
    stdt = mix.std(dim=(1, 2), keepdim=True)
    xt = (mix - meant) / (1e-5 + stdt)
    return Spec(z, x, xt, mean, std, meant, stdt)


def post(model, spec: Spec, x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """Masked iSTFT and denormalization, from the bottom of
    `HTDemucs.forward`. Returns [B, S, C, L] in float32."""
    B, _, Fq, T = spec.x.shape
    S = len(model.sources)
    length = spec.xt.shape[-1]
    x = x.float().view(B, S, -1, Fq, T) * spec.std[:, None] + spec.mean[:, None]
    # The masking and iSTFT are complex, so on MPS they run on the CPU, as in
    # `HTDemucs.forward`.
    zout = model._mask(spec.z, x.cpu() if x.device.type == "mps" else x)
    x = model._ispec(zout, length).to(spec.xt.device)
    xt = xt.float().view(B, S, -1, length) * spec.stdt[:, None] + spec.meant[:, None]
    return xt + x


class Core(nn.Module):
    """The network between `pre` and `post`: both encoders, the cross
    transformer and both decoders. Real tensors in, real tensors out, and no
    data-dependent control flow, so it compiles and exports cleanly.

    Takes x [B, 2C, Fr, T] and xt [B, C, L] from `pre`; returns
    x [B, S*2C, Fr, T] and xt [B, S*C, L] for `post`.
    """

    def __init__(self, model):
        super().__init__()
        _check_supported(model)
        self.model = model

    def forward(self, x: torch.Tensor, xt: torch.Tensor):
        m = self.model
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # lengths to remove padding with, freq.
        lengths_t = []  # lengths to remove padding with, time.
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    # Just the first conv: the branches now line up and merge.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre_ = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    xt, _ = tdec(pre_[:, :, 0], None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)
        return x, xt


def forward(model, core, mix: torch.Tensor) -> torch.Tensor:
    """`HTDemucs.forward` for a batch of training-length chunks, with the
    network run by `core` (a `Core`, or anything that behaves like one)."""
    spec = pre(model, mix)
    x, xt = core(spec.x, spec.xt)
    return post(model, spec, x, xt)


def frames(model) -> int:
    """Spectrogram frames for one chunk, which `Core`'s input shape needs."""
    return int(math.ceil(segment_length(model) / model.hop_length))


def example_inputs(model, batch: int, device="cpu", dtype=torch.float32):
    """Inputs shaped like `pre`'s output, for tracing and export."""
    C = model.audio_channels
    x = torch.randn(batch, 2 * C, model.nfft // 2, frames(model), device=device, dtype=dtype)
    xt = torch.randn(batch, C, segment_length(model), device=device, dtype=dtype)
    return x, xt


__all__ = [
    "Core",
    "Spec",
    "example_inputs",
    "forward",
    "frames",
    "load",
    "post",
    "pre",
    "segment_length",
]
