#!/bin/zsh
# Upload splits under simulated conditions, 3 each, against the new bench.
S=${WORK:-.}
cd "$(dirname "$0")"
MP3=$S/songs/musdb-testset.mp3        # 245 s, 5.9 MB
BIG=$S/songs/cristina-vane-so-easy.mp3  # 254 s, 10.2 MB
WAV=$S/songs/cristina-vane-so-easy.wav  # 254 s, 44.8 MB

run() { # label, song, env...
  local label=$1 song=$2; shift 2
  for i in 1 2 3; do env "$@" node run.mjs new $song "m:$label" 2>/dev/null | cut -c1-160; done
}

run fast           $MP3
run fast-10mb      $BIG
run home-50/20     $BIG DOWN_MBPS=50 UP_MBPS=20 RTT_MS=20
run slowhome-25/5  $BIG DOWN_MBPS=25 UP_MBPS=5 RTT_MS=40
run 4g-20/8        $BIG DOWN_MBPS=20 UP_MBPS=8 RTT_MS=60
run poor-5/1.5     $MP3 DOWN_MBPS=5 UP_MBPS=1.5 RTT_MS=150 WAIT_MS=400000
run far-rtt200     $BIG RTT_MS=200
run quick-pick     $BIG DOWN_MBPS=50 UP_MBPS=20 RTT_MS=20 PICK_MS=500
run slow-pick      $BIG DOWN_MBPS=50 UP_MBPS=20 RTT_MS=20 PICK_MS=5000
run phone-4g-cpu4x $BIG DOWN_MBPS=20 UP_MBPS=8 RTT_MS=60 CPU_X=4 MOBILE=1
run wav-home       $WAV DOWN_MBPS=50 UP_MBPS=20 RTT_MS=20
