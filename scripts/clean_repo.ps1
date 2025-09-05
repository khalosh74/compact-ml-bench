param(
  [switch] $Aggressive   = $true,   # default ON: remove runs/ and outputs/
  [switch] $RemoveData   = $false,  # keep dataset by default
  [switch] $DryRun       = $false,  # preview only
  [switch] $VerboseLog   = $true
)

function _rm([string]$path){
  if (Test-Path $path) {
    if ($DryRun) { Write-Host "[DRY] would remove $path" -ForegroundColor Yellow }
    else { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $path }
  }
}

# What we clean
$targets = @(
  "outputs",            # CSV/plots/benches
  "runs",               # checkpoints / TorchScript
  "**/__pycache__",     # python caches
  "*.log",              # logs
  ".pytest_cache",      # pytest cache
  ".ipynb_checkpoints"  # notebooks cache
)

if ($Aggressive) {
  foreach($t in $targets){
    Get-ChildItem -Path . -Recurse -Force -Filter $t -ErrorAction SilentlyContinue | ForEach-Object {
      _rm $_.FullName
    }
    # also top-level matches
    if (Test-Path $t) { _rm $t }
  }
}

if ($RemoveData) {
  if (Test-Path "data") { _rm "data" }
}

# Recreate empty artifact dirs
if (-not $DryRun) {
  New-Item -ItemType Directory -Force -Path outputs, runs | Out-Null
}

if ($VerboseLog) {
  Write-Host "[CLEAN] Aggressive=$Aggressive RemoveData=$RemoveData DryRun=$DryRun" -ForegroundColor Cyan
  if (Get-Command git -ErrorAction SilentlyContinue) {
    git status --porcelain
  }
}
