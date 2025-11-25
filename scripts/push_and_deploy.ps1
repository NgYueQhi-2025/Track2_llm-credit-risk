<#
Simple helper to commit and push local changes to `main` and open the Streamlit app.
Usage: 
  powershell -ExecutionPolicy Bypass -File .\scripts\push_and_deploy.ps1 -Message "My commit message"
If you omit -Message you will be prompted.
#>
param(
    [string]$Message = ""
)

# Ensure script runs from repo root (scripts folder is inside repo)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path (Join-Path $scriptDir "..")

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found in PATH. Install git or run this script from an environment where git is available."
    exit 1
}

if (-not $Message) {
    $Message = Read-Host "Commit message (leave empty to use default)"
    if (-not $Message) { $Message = "Update: welcome guide and UI" }
}

Write-Host "Staging changes..."
git add .

# show what will be committed
$porcelain = git status --porcelain
if (-not $porcelain) {
    Write-Host "No changes to commit." -ForegroundColor Yellow
} else {
    Write-Host "Committing changes..."
    git commit -m "$Message"
}

# Ensure we are on main branch
$curBranch = git rev-parse --abbrev-ref HEAD
if ($curBranch -ne 'main') {
    Write-Host "Switching to 'main' branch..."
    git checkout main
    Write-Host "Merging feature branch into main..."
    # merge current branch (assumes feature branch still exists locally)
    git merge --no-ff $curBranch -m "Merge $curBranch into main: $Message"
}

# create VERSION file with current short SHA
try {
    $sha = git rev-parse --short HEAD
    Set-Content -Path VERSION -Value $sha -Encoding UTF8
    git add VERSION
    git commit -m "Add VERSION $sha" -q
} catch {
    Write-Host "Warning: could not write VERSION file: $_" -ForegroundColor Yellow
}

Write-Host "Pushing to origin/main..."
git push origin main

# Open the Streamlit app URL in default browser (change if needed)
$streamlitUrl = 'https://track2llm-credit-risk-9qhysbrs3gyvv34fxxvb2p.streamlit.app/'
Write-Host "Opening Streamlit app: $streamlitUrl"
Start-Process $streamlitUrl

Write-Host "Done. Streamlit Cloud should start a deploy automatically. Check the app logs if the UI does not update."