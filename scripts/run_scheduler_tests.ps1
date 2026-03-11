<#
.SYNOPSIS
    Generation Scheduler Test Runner (Windows PowerShell)

.DESCRIPTION
    Runs all scheduler-related tests (unit + integration) with configurable options.
    This is a convenience wrapper around the Python test runner.

.PARAMETER Mode
    Run tests for a specific mode: none, internal, force, cross, edge, slow

.PARAMETER Slow
    Include slow tests (sets RUN_SLOW=1)

.PARAMETER UnitOnly
    Run only unit tests

.PARAMETER IntegrationOnly
    Run only integration (black-box) tests

.PARAMETER Verbose
    Enable verbose output

.PARAMETER FailFast
    Stop on first failure

.PARAMETER HtmlReport
    Generate an HTML test report

.EXAMPLE
    .\scripts\run_scheduler_tests.ps1
    # Run all quick tests

.EXAMPLE
    .\scripts\run_scheduler_tests.ps1 -Mode force
    # Run only FORCE mode tests

.EXAMPLE
    .\scripts\run_scheduler_tests.ps1 -Slow -HtmlReport
    # Run all tests including slow ones, generate HTML report

.EXAMPLE
    .\scripts\run_scheduler_tests.ps1 -UnitOnly -Verbose
    # Run only unit tests with verbose output
#>

param(
    [ValidateSet("none", "internal", "force", "cross", "edge", "slow")]
    [string]$Mode,

    [switch]$Slow,
    [switch]$UnitOnly,
    [switch]$IntegrationOnly,
    [switch]$Verbose,
    [switch]$FailFast,
    [switch]$HtmlReport,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Navigate to repo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Generation Scheduler — Test Runner (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Build arguments for the Python runner
$PythonArgs = @("scripts/run_scheduler_tests.py")

if ($Mode) {
    $PythonArgs += "--mode"
    $PythonArgs += $Mode
}
if ($Slow) { $PythonArgs += "--slow" }
if ($UnitOnly) { $PythonArgs += "--unit-only" }
if ($IntegrationOnly) { $PythonArgs += "--integration-only" }
if ($Verbose) { $PythonArgs += "--verbose" }
if ($FailFast) { $PythonArgs += "--failfast" }
if ($HtmlReport) { $PythonArgs += "--html-report" }
if ($DryRun) { $PythonArgs += "--dry-run" }

Write-Host "Running: python $($PythonArgs -join ' ')" -ForegroundColor Yellow
Write-Host ""

# Execute
python @PythonArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "  ALL TESTS PASSED" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  SOME TESTS FAILED (exit code: $ExitCode)" -ForegroundColor Red
}

exit $ExitCode
