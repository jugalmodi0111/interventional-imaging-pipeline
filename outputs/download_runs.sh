#!/usr/bin/env bash
# Pull each Kaggle run's full output (incl. mp4) into its own folder.
# Stenosis = your token account (jugalmodi0111) -> works now.
# Coronary + catheter = other accounts -> make those kernels Public first, or use their KAGGLE_KEY.
set -uo pipefail

BASE="/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/outputs"

STENOSIS_REF="jugalmodi0111/stenosis"
CORONARY_REF="jugalmodipoiro/coronary"
CATHETER_REF="jugalmodipesurr/catheter"

pull () {
  local ref="$1" dir="$2"
  echo "=== $ref ==="
  local st; st=$(kaggle kernels status "$ref" 2>&1 | tr -d '\r')
  echo "  status: $st"
  if echo "$st" | grep -qi complete; then
    mkdir -p "$dir"
    kaggle kernels output "$ref" -p "$dir" && echo "  downloaded -> $dir"
  elif echo "$st" | grep -qiE 'denied|private|permission'; then
    echo "  BLOCKED: private + wrong account. Make it Public, or prefix with KAGGLE_USERNAME=/KAGGLE_KEY="
  else
    echo "  not finished yet — re-run this script when it says complete"
  fi
}

pull "$STENOSIS_REF" "$BASE/run_stenosis_honest"
pull "$CORONARY_REF" "$BASE/run_coronary"
pull "$CATHETER_REF" "$BASE/run_catheter_honest"
echo "done."
