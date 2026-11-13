#!/usr/bin/env bash
# VulnReview: Reproduce everything end-to-end
# Usage: bash scripts/run_all.sh [--quick]
set -euo pipefail

QUICK=0
for arg in "$@"; do
  [[ "$arg" == "--quick" ]] && QUICK=1
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo " VulnReview Pipeline"
echo " Quick mode: $QUICK"
echo "============================================================"

echo "[1/6] Data preparation..."
if [[ $QUICK -eq 1 ]]; then
  python -m vulnreview.data_prep --quick
else
  python -m vulnreview.data_prep
fi

echo "[2/6] Running baselines..."
python -m vulnreview.baselines

echo "[3/6] Training models..."
if [[ $QUICK -eq 1 ]]; then
  python -m vulnreview.train --protocol P1 --model B4 --epochs 1 --seeds 0 --quick
else
  python -m vulnreview.train --protocol P1 --model B3 --seeds 0 1 2
  python -m vulnreview.train --protocol P1 --model B4 --seeds 0 1 2
  python -m vulnreview.train --protocol P1 --model B4 --ablation no_cwe --seeds 0 1 2
  python -m vulnreview.train --protocol P1 --model B4 --ablation no_sev --seeds 0 1 2
  python -m vulnreview.train --protocol P2 --model B3 --seeds 0 1 2
  python -m vulnreview.train --protocol P2 --model B4 --seeds 0 1 2
  python -m vulnreview.train --protocol P3 --model B3 --seeds 0 1 2
  python -m vulnreview.train --protocol P3 --model B4 --seeds 0 1 2
fi

echo "[4/6] Evaluating..."
python -m vulnreview.evaluate --summarize

echo "[5/6] Generating explanations..."
python -m vulnreview.explain --protocol P1 --max-samples 300

echo "[6/6] Running tests..."
pytest tests/ -v

echo "============================================================"
echo " ALL DONE. See results/results_summary.md"
echo "============================================================"
