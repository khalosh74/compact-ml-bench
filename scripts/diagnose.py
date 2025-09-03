import os, sys, json, platform, subprocess
rep = {}
rep["python"] = sys.version.replace("\n"," ")
rep["platform"] = platform.platform()
def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)[:2000]
    except subprocess.CalledProcessError as e:
        return f"ERR({e.returncode}): {e.output[:1000]}"
rep["which_python3"] = sh("which python3").strip()
rep["pip_V"] = sh("pip -V").strip()
rep["pip_list_torch"] = sh("pip list | grep -i torch")
try:
    import torch, torchvision
    rep["torch_version"] = torch.__version__
    rep["torchvision_version"] = getattr(torchvision, "__version__", "unknown")
    rep["cuda_available"] = torch.cuda.is_available()
except Exception as e:
    rep["import_error"] = str(e)
rep["uname"] = sh("uname -a")
rep["cpuinfo"] = sh("grep -m1 'model name' /proc/cpuinfo || true")
rep["meminfo"] = sh("head -n1 /proc/meminfo || true")
print(json.dumps(rep, indent=2))
