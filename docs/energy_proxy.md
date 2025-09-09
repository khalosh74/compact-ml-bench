# Energy proxy (GPU) — method & caveats

This project reports a **proxy** for GPU energy during benchmarks by sampling

vidia-smi --query-gpu=power.draw --format=csv,noheader,nounits at ~100 ms intervals
in a background thread and **integrating (power × time)** across the measurement window.

**Why a proxy?**
- Works on Windows and Linux without NVML bindings or admin privileges.
- Simple and portable; good for *relative* comparisons on the same machine.

**Caveats / limitations:**
- 100 ms sampling quantizes short spikes; absolute Joules are approximate.
- Shared GPU activity (other processes) will pollute readings.
- On systems without 
vidia-smi or NVIDIA GPUs, the field is recorded as 
ull.
- CPU energy is **not** measured (left 
ull).

**Best practice:**
- Run on an otherwise idle machine.
- Compare **relative** energy across variants on the same hardware and driver.
- Treat the value as indicative, not as a precise power meter.

_Last updated: 2025-09-09 19:53:09Z_
