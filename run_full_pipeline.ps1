param(
  [string]$Config = "configs/default.yaml"
)
python scripts/run.py full --config $Config

function StopIfError($msg){
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 2
  }
}

Write-Host "`n=== FINAL PIPELINE START ===" -ForegroundColor Cyan

# 0) Optional clean (keeps dataset)
if ($CleanFirst) {
  if (Test-Path "scripts/clean_repo.ps1") {
    Write-Host "[0/10] Clean old artifacts (DryRun=false)" -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File "scripts/clean_repo.ps1"
    StopIfError "Clean failed."
  } else {
    Write-Host "[WARN] scripts/clean_repo.ps1 not found; skipping clean." -ForegroundColor Yellow
  }
}

# 1) Environment check
Write-Host "`n[1/10] Environment check" -ForegroundColor Cyan
python scripts/status.py
python scripts/diagnose.py
StopIfError "Env check failed."

# 2) Data (idempotent)
Write-Host "`n[2/10] Ensure $Dataset is present in $DataDir" -ForegroundColor Cyan
python scripts/download_data.py --dataset $Dataset --out $DataDir
StopIfError "Dataset step failed."

# 3) Train baseline
Write-Host "`n[3/10] Train baseline: $Model for $Epochs epochs (batch=$Batch, workers=$Workers)" -ForegroundColor Cyan
python scripts/train.py --model $Model --epochs $Epochs --batch-size $Batch --seed $Seed --num-workers $Workers --out "runs/$OutBase"
StopIfError "Training failed."
if (!(Test-Path "runs/$OutBase/best.pt")) { Write-Host "[ERR] Missing best.pt" -ForegroundColor Red; exit 2 }
Write-Host "[OK] best.pt -> runs/$OutBase/best.pt" -ForegroundColor Green

# 4) Benchmark baseline (GPU/CPU)
Write-Host "`n[4/10] Benchmark baseline (GPU/CPU)" -ForegroundColor Cyan
python -u scripts/bench.py --checkpoint "runs/$OutBase/best.pt" --device gpu --warmup $Warmup --repeat $Repeat --verbose 1
StopIfError "GPU bench baseline failed."
$env:MKL_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"
python -u scripts/bench.py --checkpoint "runs/$OutBase/best.pt" --device cpu --warmup $Warmup --repeat $Repeat --threads 1 --out ("outputs/bench_{0}_cpu.json" -f $OutBase)
StopIfError "CPU bench baseline failed."

# 5) Structured pruning + short fine-tune (exports TorchScript)
Write-Host "`n[5/10] Structured pruning $PruneRatio and fine-tune $PruneEpochs epochs" -ForegroundColor Cyan
pip show torch-pruning >$null 2>&1; if ($LASTEXITCODE -ne 0) { pip install torch-pruning==1.6.0; StopIfError "torch-pruning install failed." }
python scripts/structured_prune.py --checkpoint "runs/$OutBase/best.pt" --ratio $PruneRatio --epochs $PruneEpochs --batch-size $Batch --num-workers $Workers --out "runs/$PrunedName"
StopIfError "Structured prune failed."
if (!(Test-Path "runs/$PrunedName/structured.ts")) { Write-Host "[ERR] Missing TorchScript: runs/$PrunedName/structured.ts" -ForegroundColor Red; exit 2 }
Write-Host "[OK] TorchScript -> runs/$PrunedName/structured.ts" -ForegroundColor Green

# 6) Benchmark pruned TorchScript (GPU & CPU)
Write-Host "`n[6/10] Benchmark pruned TorchScript (GPU/CPU)" -ForegroundColor Cyan
python scripts/bench_ts.py --artifact "runs/$PrunedName/structured.ts" --device gpu --warmup $Warmup --repeat $Repeat --verbose 1
StopIfError "TS GPU bench failed."
python scripts/bench_ts.py --artifact "runs/$PrunedName/structured.ts" --device cpu --warmup $Warmup --repeat $Repeat --out ("outputs/bench_{0}_cpu.json" -f $PrunedName)
StopIfError "TS CPU bench failed."

# 7) Summarize & plots
Write-Host "`n[7/10] Summarize & plots" -ForegroundColor Cyan
python scripts/summarize.py
StopIfError "Summarize failed."
python scripts/make_plots.py
StopIfError "Plot generation failed."
Get-ChildItem outputs\results.csv, outputs\plot_acc_vs_size.png, outputs\plot_acc_vs_latency.png | Format-Table Name,Length


# 9) Optional: launch Streamlit demo
 Optional: launch Streamlit demo
if ($OpenStreamlit -and (Test-Path "demo_app.py")) {
  Write-Host "`n[9/10] Launching Streamlit demo (Ctrl+C to stop)" -ForegroundColor Cyan
  streamlit run demo_app.py -- --checkpoint "runs/$OutBase/best.pt" --ts "runs/$PrunedName/structured.ts"
}

Write-Host "`n[10/10] DONE ?  Results in 'outputs' + artifacts in 'runs' + $zip" -ForegroundColor Green

