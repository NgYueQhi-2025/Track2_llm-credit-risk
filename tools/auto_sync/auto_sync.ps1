param(
    [int]$IntervalMinutes = 5,
    [string]$Remote = 'origin',
    [string]$Branch = 'main'
)

Set-StrictMode -Version Latest

# Change to the repository root (script location assumed under tools/auto_sync)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Set-Location ..\..

Write-Output "Starting auto-sync: remote=$Remote branch=$Branch interval=${IntervalMinutes}m"

function Get-ShortHash($ref) {
    try {
        return (& git rev-parse --short $ref).Trim()
    } catch {
        return $null
    }
}

while ($true) {
    try {
        Write-Output "[AutoSync] Fetching $Remote..."
        & git fetch $Remote --prune

        $local = Get-ShortHash HEAD
        $remoteRef = "$Remote/$Branch"
        $remoteHead = Get-ShortHash $remoteRef

        if (!$remoteHead) {
            Write-Output "[AutoSync] Remote ref $remoteRef not found. Will retry in $IntervalMinutes minutes."
        } elseif ($local -ne $remoteHead) {
            Write-Output "[AutoSync] Remote changed: local=$local remote=$remoteHead"
            # Try a fast-forward pull only to avoid merge conflicts.
            try {
                Write-Output "[AutoSync] Attempting fast-forward pull from $remoteRef..."
                & git pull --ff-only $Remote $Branch
                if ($LASTEXITCODE -eq 0) {
                    $now = Get-Date -Format o
                    Write-Output "[AutoSync] Pulled changes successfully at $now"
                    Add-Content -Path .git\auto_sync.log -Value "Pulled $remoteHead at $now"
                } else {
                    Write-Warning "[AutoSync] Fast-forward pull failed (non-zero exit). Manual intervention required. Stopping auto-sync."
                    break
                }
            } catch {
                Write-Warning "[AutoSync] Pull failed: $_. Exception. Manual resolution required. Stopping auto-sync."
                break
            }
        } else {
            Write-Output "[AutoSync] Up to date (HEAD $local)."
        }
    } catch {
        Write-Warning "[AutoSync] Unexpected error: $_"
    }

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}

Write-Output "Auto-sync stopped." 
