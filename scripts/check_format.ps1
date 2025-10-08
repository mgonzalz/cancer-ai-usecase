<#
.SYNOPSIS
    Checks Python code formatting and linting in the specified folder or file.

.DESCRIPTION
    This script runs static code quality tools without modifying any files.
    It performs checks using:
        - isort: validates import order.
        - black: verifies code style compliance.
        - flake8: runs linting checks (E501 long-line warnings are ignored).

.PARAMETER Path
    The folder or file to check. Defaults to the current directory ('.').

.EXAMPLE
    # Check formatting and linting in the 'tests' folder
    .\scripts\check_format.ps1 -Path tests\

.EXAMPLE
    # Check a single file for format and lint issues
    .\scripts\check_format.ps1 -Path tests\test_gpu_tensorflow.py

.OUTPUTS
    Exit code 0 if no issues are found; 1 otherwise.

.NOTES
    Requires black, isort, and flake8 to be installed in the active Python environment.
    Can be used inside Poetry-managed environments or standalone virtual environments (.venv).
#>

param(
    [string]$Path = "."
)

Write-Host "Checking isort on: $Path"
isort --check-only $Path; $exit_isort = $LASTEXITCODE

Write-Host "Checking black on: $Path"
black --check $Path;      $exit_black = $LASTEXITCODE

Write-Host "Checking flake8 on: $Path"
flake8 --max-line-length=100 $Path;             $exit_flake = $LASTEXITCODE

if ($exit_isort -eq 0 -and $exit_black -eq 0 -and $exit_flake -eq 0) {
    Write-Host "`nAll good on '$Path'." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nIssues found (isort=$exit_isort, black=$exit_black, flake8=$exit_flake) in '$Path'." -ForegroundColor Red
    exit 1
}
