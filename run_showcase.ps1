[CmdletBinding()]
param(
  # ---- Core training/bench params ----
  [int]    $Epochs      = 40,
  [int]    $Batch       = 1024,
  [int]    $Workers     = 16,
  [int]    $Warmup      = 60,
  [int]    $Repeat      = 600,
  [string] $Model       = "resnet18",
  [int]    $Seed        = 42,

  # ---- Run names ----
  [string] $OutBase     = "baseline_40e",
  [string] $PrunedName  = "resnet18_struct30",

  # ---- Pruning ----
  [double] $PruneRatio  = 0.3,
  [int]    $PruneEpochs = 10,

  # ---- Data ----
  [string] $Dataset     = "cifar10",
  [string] $DataDir     = "data",

  # ---- Extras ----
  [switch] $OpenStreamlit  # launch the Streamlit demo at the end
)

function StopIfError($msg){
  if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 2 }
}

# Detect CUDA for benches
$cudaAvail = & python -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')"
$doGPU = ($cudaAvail -eq '1')

New-Item -ItemType Directory -Force -Path outputs | Out-Null
New-Item -ItemType Directory -Force -Path runs    | Out-Null

Write-Host "`n[0/9] Env check" -ForegroundColor Cyan
python scripts/status.py
python scripts/diagnose.py
StopIfError "Env check failed."

Write-Host "`n[1/9] Ensure dataset" -ForegroundColor Cyan
python scripts/download_data.py --dataset $Dataset --out $DataDir
StopIfError "Dataset download failed."

Write-Host "`n[2/9] Full baseline training ($Epochs e, batch $Batch, workers $Workers)" -ForegroundColor Cyan
.\run_pipeline.ps1 -Epochs $Epochs -Batch $Batch -Workers $Workers -Warmup $Warmup -Repeat $Repeat `
  -Model $Model -Seed $Seed -OutBase $OutBase -PrunedName $PrunedName -PruneRatio $PruneRatio -PruneEpochs $PruneEpochs `
  -SkipTrain:$false -SkipBench:$false -SkipPrune:$false -SkipPlots:$false
StopIfError "Pipeline run failed."

if (!(Test-Path "runs/$OutBase/best.pt")) { Write-Host "[ERR] Missing runs/$OutBase/best.pt" -ForegroundColor Red; exit 2 }
if (!(Test-Path "runs/$PrunedName/structured.ts")) { Write-Host "[ERR] Missing runs/$PrunedName/structured.ts" -ForegroundColor Red; exit 2 }

Write-Host "`n[3/9] Re-bench baseline GPU into CSV for consistency" -ForegroundColor Cyan
if ($doGPU) {
  python -u scripts/bench.py --checkpoint "runs/$OutBase/best.pt" --device gpu --warmup $Warmup --repeat $Repeat --out "outputs/bench_${OutBase}_gpu.json"
  StopIfError "Baseline GPU bench failed."
} else {
  Write-Host "[INFO] CUDA not available; skipping baseline GPU bench." -ForegroundColor Yellow
}

Write-Host "`n[4/9] Refresh summary & plots" -ForegroundColor Cyan
python scripts/summarize.py
StopIfError "Summarize failed."
python scripts/make_plots.py
StopIfError "Plot generation failed."

Write-Host "`n[5/9] Create a short README-style summary (Markdown)" -ForegroundColor Cyan
$md = @"
# Compact-ML Bench — Final Showcase

**Baseline:** $Model ($Epochs epochs)  
**Pruned:** structured channels = $([math]::Round($PruneRatio*100,0))% → TorchScript

Artifacts:
- \`runs\$OutBase\best.pt\`
- \`runs\$PrunedName\structured.ts\`
- \`outputs\results.csv\`
- \`outputs\plot_acc_vs_size.png\`
- \`outputs\plot_acc_vs_latency.png\`

Reproduce:
\`\`\`powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_pipeline.ps1 -Epochs $Epochs -Batch $Batch -Workers $Workers -Warmup $Warmup -Repeat $Repeat `
  -OutBase "$OutBase" -PrunedName "$PrunedName" -PruneRatio $PruneRatio -PruneEpochs $PruneEpochs
\`\`\`
"@
$md | Set-Content -Encoding UTF8 outputs\README_showcase.md

Write-Host "`n[6/9] Package key files (showcase.zip)" -ForegroundColor Cyan
$toZip = @(
  "outputs\results.csv",
  "outputs\plot_acc_vs_size.png",
  "outputs\plot_acc_vs_latency.png",
  "outputs\README_showcase.md",
  "runs\$OutBase\best.pt",
  "runs\$PrunedName\structured.ts"
) | ?{ Test-Path $_ }
if (Test-Path showcase.zip) { Remove-Item showcase.zip -Force }
if ($toZip.Count -gt 0) { Compress-Archive -Force -Path $toZip -DestinationPath showcase.zip }

Write-Host "`n[7/9] Artifact list" -ForegroundColor Cyan
Get-ChildItem -Recurse runs -Include *.pt,*.pth,*.ts | Format-Table FullName,Length
Get-ChildItem outputs\results.csv, outputs\plot_acc_vs_* | Format-Table Name,Length

Write-Host "`n[8/9] Optional: launch Streamlit demo" -ForegroundColor Cyan
if ($OpenStreamlit) {
  pip show streamlit >$null 2>&1; if ($LASTEXITCODE -ne 0) { pip install streamlit; StopIfError "Failed to install streamlit." }
  Start-Process powershell -ArgumentList 'streamlit run streamlit_app.py'
  Write-Host "[INFO] Streamlit launched in a new window." -ForegroundColor Green
} else {
  Write-Host "[INFO] Skipping Streamlit launch. Use: streamlit run streamlit_app.py" -ForegroundColor Yellow
}

Write-Host "`n[9/9] DONE ✅  Results in 'outputs', artifacts in 'runs', bundle 'showcase.zip' ready." -ForegroundColor Green
