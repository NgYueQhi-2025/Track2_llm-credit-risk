<#
PowerShell helper to run the Track2 LLM Credit Risk Streamlit app.
Usage: From the repository root run: .\run_track2.ps1
This script will try to activate a venv at .\.venv and then run Streamlit
#>
try {
    $venvActivate = Join-Path -Path (Get-Location) -ChildPath ".venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        Write-Host "Activating virtual environment..."
        & $venvActivate
    } else {
        Write-Host "No .venv found at $venvActivate. Ensure your Python environment is activated manually if needed."
    }
} catch {
    Write-Warning "Failed to auto-activate venv: $_"
}

$appPath = Join-Path -Path (Get-Location) -ChildPath "Track2_llm-credit-risk\backend\app.py"
if (-not (Test-Path $appPath)) {
    Write-Error "Track2 app not found at: $appPath"
    exit 1
}

Write-Host "Starting Streamlit for Track2 app: $appPath"
streamlit run "$appPath"
