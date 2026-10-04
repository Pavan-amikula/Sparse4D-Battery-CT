param([string]$Python = "C:\Users\paam25\MLProjects\Sparse4D_Project\.venv\Scripts\python.exe", [int]$Port = 8765)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Python)) { throw "Python environment not found: $Python. Pass -Python with your GPU environment's python.exe path." }
Set-Location $PSScriptRoot
& $Python -c "import numpy, torch, astra, tifffile; print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available()); print('ASTRA:',astra.__version__); print('NumPy:',numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "A required dependency is missing. Use the same Python environment as your verified experiments. This launcher does not replace packages." }
& $Python "$PSScriptRoot\app.py" --port $Port
