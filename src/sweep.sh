#!/bin/bash
# sweep.sh — small hyperparameter sweep for the fine-tune; keeps the best model
# (selected by validation macro-F1). Writes outputs/sweep_results.json.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

# "lr epochs batch patience weighted(0/1)"
CONFIGS=(
  "2e-5 14 16 4 0"
  "5e-5 14 16 4 0"
  "4e-5 16 16 4 0"
  "5e-5 16 16 4 1"
  "3e-5 16 16 4 1"
)

best=-1; best_cfg=""; rows="["
for cfg in "${CONFIGS[@]}"; do
  read lr ep bs pa wt <<< "$cfg"
  wflag=""; [ "$wt" = "1" ] && wflag="--class_weights"
  echo ">>> training lr=$lr epochs=$ep batch=$bs patience=$pa weighted=$wt"
  $PY src/train.py --lr "$lr" --epochs "$ep" --batch "$bs" --patience "$pa" $wflag >/tmp/sweep_run.log 2>&1 || { echo "  run failed"; tail -3 /tmp/sweep_run.log; continue; }
  val=$($PY -c "import json;print(json.load(open('outputs/train_log.json'))['best_val_macro_f1'])")
  bep=$($PY -c "import json;print(json.load(open('outputs/train_log.json'))['best_epoch'])")
  echo "    -> best_val_macro_f1=$val (epoch $bep)"
  rows="$rows{\"lr\":\"$lr\",\"epochs\":$ep,\"batch\":$bs,\"weighted\":$wt,\"val_macro_f1\":$val,\"best_epoch\":$bep},"
  better=$($PY -c "print(1 if $val > $best else 0)")
  if [ "$better" = "1" ]; then
    best=$val; best_cfg="$cfg"
    rm -rf model_best; cp -r model model_best
  fi
done
rows="${rows%,}]"

# restore the best model
rm -rf model; mv model_best model
$PY -c "import json; json.dump({'best_cfg':'$best_cfg','best_val_macro_f1':$best,'runs':$rows}, open('outputs/sweep_results.json','w'), indent=2)"
echo "=== BEST: $best_cfg  val_macro_f1=$best ==="
