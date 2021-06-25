#!/bin/bash
WAV="$1"
if [ -z "$WAV" ]; then
    echo "Usage: $0 OUTPUT.WAV" >&2
    exit 1
fi
rm -f "$WAV"

# Get sink monitor:
MONITOR=$(pactl list | egrep -A2 '^(\*\*\* )?Source #' | \
    grep 'Name: .*\.monitor$' | awk '{print $NF}' | tail -n1)
MONITOR="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
echo "set-source-mute ${MONITOR} false" | pacmd >/dev/null
echo $MONITOR

# Record it raw, and convert to a wav
echo "Recording to $WAV ..."
parec -d "$MONITOR" | sox -r 44.1k -e signed -b 16 -c 2 -t raw - "$WAV"
