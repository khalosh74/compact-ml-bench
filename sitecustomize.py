# Python auto-import hook: if present in project root, it will be imported on startup
import os
try:
    from utils.determinism import set_run_mode_from_env
    if os.getenv("ML_MODE"):
        set_run_mode_from_env()
except Exception:
    # Never block execution due to determinism tuning
    pass
