<#
.SYNOPSIS
    Automatically fixes Python code formatting and import order in the specified folder or file.

.DESCRIPTION
    This script formats and cleans Python code in place.
    It performs the following actions:
        - isort: sorts and groups imports.
        - black: reformats Python files to meet code style standards.
        - flake8: runs linting checks after formatting (E501 long-line warnings are ignored).

.PARAMETER Path
    The folder or file to fix. Defaults to the current directory ('.').

.EXAMPLE
    # Automatically fix code formatting and imports in the 'tests' folder
    .\scripts\fix_format.ps1 -Path tests\

.EXAMPLE
    # Fix and verify a single Python file
    .\scripts\fix_format.ps1 -Path tests\test_gpu_tensorflow.py

.OUTPUTS
    Exit code 0 if formatting and linting pass after fixes; 1 if lint issues remain.

.NOTES
    Requires black, isort, and flake8 to be installed in the active Python environment.
    Compatible with Poetry or standard Python virtual environments (.venv).
#>

param(
    [string]$Path = "."
)

Write-Host "Running isort on: $Path"
isort $Path

Write-Host "Running black on: $Path"
black $Path

Write-Host "Running flake8 on: $Path"
flake8 --max-line-length=100 $Path; $exit_flake = $LASTEXITCODE

if ($exit_flake -eq 0) {
    Write-Host "`nFixed & clean on '$Path'." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nFixed formatting, but flake8 still reports issues (code=$exit_flake) on '$Path'." -ForegroundColor Yellow
    exit 1
}
