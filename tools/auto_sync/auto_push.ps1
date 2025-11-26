param(
    [string]$Remote = 'origin',
    [string]$Branch = 'main',
    [string]$Message = $("Auto-sync: $(Get-Date -Format o)" )
)

Set-StrictMode -Version Latest

# Change to repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Set-Location ..\..

Write-Output "Auto-push starting: remote=$Remote branch=$Branch"

# Check for changes
$porcelain = (& git status --porcelain) -join "`n"
if (-not $porcelain) {
    Write-Output "No changes to commit. Exiting."
    exit 0
}

Write-Output "Changes detected. Staging and committing..."
& git add -A
$commit = & git commit -m "$Message"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Commit failed or nothing to commit. Output:`n$commit"
    exit 1
}

Write-Output "Pushing to $Remote/$Branch..."
& git push $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Push failed. Exit code: $LASTEXITCODE"
    exit 1
}

Write-Output "Push completed successfully."
