[CmdletBinding()]
param(
  # ---- Core training/bench params ----
  [int]    $Epochs      = 40,
  [int]    $Batch       = 1024,
  [int]    $Workers     = 16,
  [int]    $Warmup      = 40,
  [int]    $Repeat      = 400,
  [string] $Model       = "resnet18",
  [int]    $Seed        = 42,

  # ---- Names for run folders ----
  [string] $OutBase     = "baseline_40e",
  [string] $PrunedName  = "resnet18_struct30",

  # ---- Pruning params ----
  [double] $PruneRatio  = 0.3,
  [int]    $PruneEpochs = 8,

  # ---- Data ----
  [string] $Dataset     = "cifar10",
  [string] $DataDir     = "data",

  # ---- Switches ----
  [switch] $CpuOnly,
  [switch] $SkipTrain,
  [switch] $SkipBench,
  [switch] $SkipPrune,
  [switch] $SkipPlots
)

function StopIfError($msg){
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 2
  }
}

# Detect CUDA availability (for optional GPU benches)
$cudaAvail = & python -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')"
$doGPU = (-not $CpuOnly) -and ($cudaAvail -eq '1')

New-Item -ItemType Directory -Force -Path outputs | Out-Null
New-Item -ItemType Directory -Force -Path runs    | Out-Null

Write-Host "`n[1/9] Environment check" -ForegroundColor Cyan
python scripts/status.py
python scripts/diagnose.py
StopIfError "Env check failed."

Write-Host "`n[2/9] Download dataset if missing" -ForegroundColor Cyan
python scripts/download_data.py --dataset $Dataset --out $DataDir
StopIfError "Dataset download failed."

# ---------------- TRAIN ----------------
if (-not $SkipTrain) {
  Write-Host "`n[3/9] Train $Model baseline ($Epochs epochs, batch $Batch, workers $Workers)" -ForegroundColor Cyan
  python scripts/train.py --model $Model --epochs $Epochs --batch-size $Batch --seed $Seed --num-workers $Workers --out "runs/$OutBase"
  StopIfError "Training failed."
  if (Test-Path "runs/$OutBase/best.pt") {
    Write-Host "[OK] best.pt -> runs/$OutBase/best.pt" -ForegroundColor Green
  } else {
    Write-Host "[ERR] Missing best.pt" -ForegroundColor Red; exit 2
  }
} else {
  Write-Host "[SKIP] Training skipped (using existing runs/$OutBase/best.pt)" -ForegroundColor Yellow
}

# ---------------- BENCH BASELINE ----------------
if (-not $SkipBench) {
  Write-Host "`n[4/9] Benchmark baseline" -ForegroundColor Cyan

  if ($doGPU) {
    python -u scripts/bench.py --checkpoint "runs/$OutBase/best.pt" --device gpu --warmup $Warmup --repeat $Repeat --verbose 1
    StopIfError "GPU bench failed."
  } else {
    Write-Host "[INFO] GPU bench skipped (CpuOnly or CUDA not available)" -ForegroundColor Yellow
  }

  # CPU bench (pin threads for stability)
  $env:MKL_NUM_THREADS = "1"
  $env:OMP_NUM_THREADS = "1"
  python -u scripts/bench.py --checkpoint "runs/$OutBase/best.pt" --device cpu --warmup $Warmup --repeat $Repeat --threads 1 --out "outputs/bench_${OutBase}_cpu.json"
  StopIfError "CPU bench failed."
} else {
  Write-Host "[SKIP] Baseline bench skipped" -ForegroundColor Yellow
}

# ---------------- STRUCTURED PRUNE + TS ----------------
if (-not $SkipPrune) {
  Write-Host "`n[5/9] Structured pruning ($([math]::Round($PruneRatio*100,0))%) + fine-tune ($PruneEpochs epochs)" -ForegroundColor Cyan
  # Ensure torch-pruning
  pip show torch-pruning >$null 2>&1
  if ($LASTEXITCODE -ne 0) {
    pip install torch-pruning==1.6.0
    StopIfError "torch-pruning install failed."
  }

  python scripts/structured_prune.py --checkpoint "runs/$OutBase/best.pt" `
    --ratio $PruneRatio --epochs $PruneEpochs --batch-size $Batch --num-workers $Workers --out "runs/$PrunedName"
  StopIfError "Structured prune failed."

  if (Test-Path "runs/$PrunedName/structured.ts") {
    Write-Host "[OK] TorchScript created -> runs/$PrunedName/structured.ts" -ForegroundColor Green
  } else {
    Write-Host "[ERR] Missing runs/$PrunedName/structured.ts" -ForegroundColor Red
    exit 2
  }

  # ---------------- BENCH PRUNED TS ----------------
  Write-Host "`n[6/9] Benchmark pruned TorchScript" -ForegroundColor Cyan
  if ($doGPU) {
    python scripts/bench_ts.py --artifact "runs/$PrunedName/structured.ts" --device gpu --warmup $Warmup --repeat $Repeat --verbose 1
    StopIfError "TS GPU bench failed."
  } else {
    Write-Host "[INFO] TS GPU bench skipped (CpuOnly or CUDA not available)" -ForegroundColor Yellow
  }
  python scripts/bench_ts.py --artifact "runs/$PrunedName/structured.ts" --device cpu --warmup $Warmup --repeat $Repeat --out "outputs/bench_${PrunedName}_cpu.json"
  StopIfError "TS CPU bench failed."
} else {
  Write-Host "[SKIP] Prune + TS + pruned benches skipped" -ForegroundColor Yellow
}

# ---------------- SUMMARIZE + PLOTS ----------------
if (-not $SkipPlots) {
  Write-Host "`n[7/9] Summarize results and make plots" -ForegroundColor Cyan
  python scripts/summarize.py
  StopIfError "Summarize failed."
  python scripts/make_plots.py
  StopIfError "Plot generation failed."
  Get-ChildItem outputs\results.csv, outputs\plot_acc_vs_size.png, outputs\plot_acc_vs_latency.png | Format-Table Name,Length
} else {
  Write-Host "[SKIP] Plots skipped" -ForegroundColor Yellow
}

Write-Host "`n[8/9] Quick artifact list" -ForegroundColor Cyan
Get-ChildItem -Recurse runs -Include *.pt,*.pth,*.ts | Format-Table FullName,Length

Write-Host "`n[9/9] Done ✅  Artifacts and results are in 'runs' and 'outputs'." -ForegroundColor Green
