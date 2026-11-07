#!/usr/bin/env bash
# Downloads the SIDD subset used by this project (~7.5 GB) from the SIDD-CC release on Hugging Face:
#   - for each of the 115 training capture instances: the first NOISY_RAW + METADATA_RAW + GT_SRGB
#   - the official benchmark: noisy RAW blocks + clean sRGB blocks + block coordinates
# Usage: bash scripts/download_sidd.sh [target_dir]   (default: ../sidd, i.e. next to the repo)
set -euo pipefail
OUT="${1:-$(dirname "$0")/../../sidd}"
B=https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/resolve/main
API=https://huggingface.co/api/datasets/MhdMohammadi/SIDD-CC/tree/main
mkdir -p "$OUT"/{raw,srgb,test}
cd "$OUT"
curl -sSL -o insts.txt "$B/Raw/train/Scene_Instances.txt"
curl -sSL -o test_scenes_info.txt "$B/Raw/test/test_scenes_info.txt"
: > urls.txt
while read -r inst; do
  [ -z "$inst" ] && continue
  f=$(curl -sS "$API/Raw/train/Data/$inst" | python3 -c "import json,sys; print(sorted(x['path'] for x in json.load(sys.stdin) if 'NOISY_RAW' in x['path'])[0])")
  sid=${inst%%_*}; idx=$(basename "$f" .MAT); idx=${idx##*_}
  echo "$B/Raw/train/Data/$inst/${sid}_NOISY_RAW_${idx}.MAT raw/${inst}_NOISY.MAT" >> urls.txt
  echo "$B/Raw/train/Data/$inst/${sid}_METADATA_RAW_${idx}.MAT raw/${inst}_META.MAT" >> urls.txt
  echo "$B/SRGB/train/Data/$inst/${sid}_GT_SRGB_${idx}.PNG srgb/${inst}_GT.PNG" >> urls.txt
done < insts.txt
echo "$B/Raw/test/BenchmarkNoisyBlocksRaw.mat test/BenchmarkNoisyBlocksRaw.mat" >> urls.txt
echo "$B/SRGB/test/BenchmarkGtBlocksSrgb.mat test/BenchmarkGtBlocksSrgb.mat" >> urls.txt
echo "$B/Raw/test/BenchmarkBlocks32.mat test/BenchmarkBlocks32.mat" >> urls.txt
xargs -P 8 -L 1 sh -c '[ -s "$1" ] || curl -sSL --retry 3 -o "$1" "$0"' < urls.txt
echo "SIDD subset ready in $OUT ($(ls raw | grep -c NOISY) training frames)"
# LPIPS backbone weights (used by evaluate.py)
python3 -c "import lpips; lpips.LPIPS(net='alex', verbose=False)" >/dev/null 2>&1 || true
