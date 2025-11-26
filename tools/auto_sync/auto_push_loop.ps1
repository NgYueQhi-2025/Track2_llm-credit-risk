param(
    [int]$IntervalSeconds = 5,
    [string]$Remote = 'origin',
    [string]$Branch = 'main'
)

Set-StrictMode -Version Latest

# Change to repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Set-Location ..\..

Write-Output "Starting auto-push loop: remote=$Remote branch=$Branch interval=${IntervalSeconds}s"

function Has-Changes {
    $p = (& git status --porcelain)
    return -not [string]::IsNullOrEmpty(($p -join "`n"))
}

while ($true) {
    try {
        if (Has-Changes) {
            Write-Output "[AutoPush] Local changes detected. Running auto_push.ps1..."
            & "$PSScriptRoot\\auto_push.ps1" -Remote $Remote -Branch $Branch
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "[AutoPush] auto_push script failed with exit code $LASTEXITCODE. Will retry later."
            }
        } else {
            Write-Output "[AutoPush] No local changes."
        }
    } catch {
        Write-Warning "[AutoPush] Unexpected error: $_"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
