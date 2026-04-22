$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = (Resolve-Path (Join-Path $scriptDir '..')).Path
$repoRoot = (Resolve-Path (Join-Path $appDir '..')).Path
$frontendHost = if ($env:FACEAI_FRONTEND_HOST) { $env:FACEAI_FRONTEND_HOST } else { '0.0.0.0' }
$backendHost = if ($env:FACEAI_BACKEND_HOST) { $env:FACEAI_BACKEND_HOST } else { '0.0.0.0' }
$backendPort = if ($env:FACEAI_BACKEND_PORT) { $env:FACEAI_BACKEND_PORT } else { '8000' }
$venvPython = Join-Path $appDir 'backend\.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
  throw "Backend virtual environment not found at $venvPython. Run: python -m venv faceai/backend/.venv and then faceai/backend/.venv/Scripts/python -m pip install -r faceai/backend/requirements.txt"
}

Write-Host 'Starting FaceAI dev servers...'
Write-Host 'Frontend: http://localhost:5173'
Write-Host "Backend:  http://localhost:$backendPort"

$frontendProcess = Start-Process -FilePath 'cmd.exe' `
  -ArgumentList '/d', '/c', "npm.cmd --prefix frontend run dev -- --host $frontendHost" `
  -WorkingDirectory $appDir `
  -NoNewWindow `
  -PassThru

$backendProcess = Start-Process -FilePath $venvPython `
  -ArgumentList @(
    '-m',
    'uvicorn',
    'app.main:app',
    '--host',
    $backendHost,
    '--port',
    $backendPort,
    '--app-dir',
    'backend'
  ) `
  -WorkingDirectory $appDir `
  -NoNewWindow `
  -PassThru

try {
  while ($true) {
    Start-Sleep -Seconds 1
    $frontendProcess.Refresh()
    $backendProcess.Refresh()

    if ($frontendProcess.HasExited -or $backendProcess.HasExited) {
      break
    }
  }
} finally {
  if ($frontendProcess -and -not $frontendProcess.HasExited) {
    Stop-Process -Id $frontendProcess.Id -Force
  }

  if ($backendProcess -and -not $backendProcess.HasExited) {
    Stop-Process -Id $backendProcess.Id -Force
  }
}

if ($frontendProcess.HasExited) {
  throw "Frontend exited with code $($frontendProcess.ExitCode)."
}

if ($backendProcess.HasExited) {
  throw "Backend exited with code $($backendProcess.ExitCode)."
}
