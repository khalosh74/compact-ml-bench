#!/usr/bin/env bash
set -Eeuo pipefail
echo "===== DIAGNOSE START ====="
echo "Python: $(python3 -V || true)"
echo "Which python: $(which python3 || true)"
echo "VENV python: $( [ -x .venv/bin/python ] && echo .venv/bin/python || echo 'N/A')"
echo "Pip: $(pip -V || true)"
echo "VENV pip: $( [ -x .venv/bin/pip ] && .venv/bin/pip -V || echo 'N/A')"
echo "pip list (torch*):"
( [ -x .venv/bin/pip ] && .venv/bin/pip list | grep -i torch ) || ( pip list | grep -i torch || true )
echo "uname: $(uname -a)"
echo "CPU info:"
grep -m1 "model name" /proc/cpuinfo || true
echo "Mem (first line):"
head -n1 /proc/meminfo || true
echo "Torch quick import:"
( .venv/bin/python - <<'PY'
try:
    import torch, torchvision
    print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
    print("torchvision:", getattr(__import__("torchvision"),"__version__", "unknown"))
except Exception as e:
    print("IMPORT_ERROR:", e)
    raise SystemExit(1)
PY
) || true
echo "===== DIAGNOSE END ====="
