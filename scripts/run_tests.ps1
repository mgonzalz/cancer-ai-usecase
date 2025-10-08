<#
.SYNOPSIS
    Runs Python unittests for a folder, a single file, or a dotted module.

.DESCRIPTION
    - If Path is a directory: runs unittest discovery on that directory.
    - If Path is a .py file: runs discovery in its parent dir with a pattern for that file.
    - Otherwise: treats Path as a dotted module (e.g., tests.test_gpu_tensorflow).

    Uses Poetry if available (poetry + poetry.lock alongside this script), otherwise uses
    the active Python (venv/global). No Invoke-Expression; arguments are passed safely.

.PARAMETER Path
    Folder, single test file (.py), or dotted module. Defaults to 'tests'.

.EXAMPLE
    .\scripts\run_tests.ps1                     # runs tests/
    .\scripts\run_tests.ps1 -Path tests\        # directory
    .\scripts\run_tests.ps1 -Path tests\test_gpu_tensorflow.py  # single file
    .\scripts\run_tests.ps1 -Path tests.test_gpu_tensorflow     # dotted module

.OUTPUTS
    Exit code 0 if all tests pass; non-zero otherwise.
#>

param(
    [string]$Path = "tests"
)

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "`nNo active Python environment detected." -ForegroundColor Red
    Write-Host "Activate .venv (.\.venv\Scripts\activate) or use Poetry (poetry shell)." -ForegroundColor Yellow
    exit 1
}

$PoetryExists   = ($null -ne (Get-Command poetry -ErrorAction SilentlyContinue))
$PoetryLockHere = Test-Path (Join-Path $PSScriptRoot 'poetry.lock')
$UsePoetry      = $PoetryExists -and $PoetryLockHere
Write-Host ("Detected environment: " + ($(if ($UsePoetry) { "Poetry" } else { "Virtualenv/Global Python" }))) -ForegroundColor DarkGray

function Run-Unittest {
    param([string[]]$Args)
    if ($UsePoetry) {
        & poetry @('run','python','-m','unittest') @Args
    } else {
        & python @('-m','unittest') @Args
    }
    return $LASTEXITCODE
}


$exit = 1
if ((Test-Path -Path $Path -PathType Leaf) -and ($Path.ToLower().EndsWith('.py'))) {
    $dir = Split-Path -Parent $Path
    $pat = Split-Path -Leaf   $Path
    Write-Host "Running unittests (single file) via discovery: dir='$dir', pattern='$pat'"
    $exit = Run-Unittest @('discover','-s', $dir, '-p', $pat)
}
elseif (Test-Path -Path $Path -PathType Container) {
    Write-Host "Running unittests (directory) via discovery: dir='$Path'"
    $exit = Run-Unittest @('discover','-s', $Path)
}
else {
    Write-Host "Running unittests (module): $Path"
    $exit = Run-Unittest @($Path)
}

if ($exit -eq 0) {
    Write-Host "`nAll tests passed successfully." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nSome tests failed. (exit code $exit)" -ForegroundColor Red
    Write-Host "If you passed a file, remember discovery needs directory + pattern." -ForegroundColor Yellow
    Write-Host "Example: python -m unittest discover -s tests -p test_gpu_tensorflow.py" -ForegroundColor Yellow
    exit 1
}
