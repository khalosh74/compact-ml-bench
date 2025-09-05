# sitecustomize.py (auto-imported by Python if this folder is on sys.path)
import os
try:
    from utils.determinism import set_run_mode_from_env
except Exception as e:
    print(f"[MODE] sitecustomize: utils/determinism not available ({e})", flush=True)
else:
    # Only act if user actually set a mode; otherwise stay quiet.
    if os.getenv("ML_MODE"):
        set_run_mode_from_env()
