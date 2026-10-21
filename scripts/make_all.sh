#!/usr/bin/env bash
# Rebuild every number, figure and the PDF from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python

echo "== smoke tests =="
$PY tests/test_smoke.py

echo "== benchmark (3 seeds) =="
$PY -m forge.bench --suite full --seeds 3 --tag final_s3

echo "== design loop (offline replay) =="
$PY -m forge.search --proposer recorded --rounds 12 --smoke-seeds 2 --full-seeds 2 --top-k 4

echo "== figures =="
$PY -m forge.plots --raw results/raw/final_s3.jsonl

echo "== report =="
$PY report/build.py

echo
echo "done → report/report.pdf"
