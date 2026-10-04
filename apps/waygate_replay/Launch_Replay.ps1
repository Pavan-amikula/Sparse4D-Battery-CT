param([string]$Python = "C:\Users\paam25\MLProjects\Sparse4D_Project\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path "$PSScriptRoot\prepared_waygate\preparation_report.json")) {
    & $Python "$PSScriptRoot\prepare_waygate.py"
    if ($LASTEXITCODE -ne 0) { throw "Preparation failed. See the message above." }
}
Write-Host "Experimental archived measured data. Geometry and detector correction remain unconfirmed."
Write-Host "Open http://127.0.0.1:8766 and choose Waygate archive replay."
& $Python "$PSScriptRoot\app.py" --port 8766
