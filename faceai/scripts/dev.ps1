$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = (Resolve-Path (Join-Path $scriptDir '..')).Path
$repoRoot = (Resolve-Path (Join-Path $appDir '..')).Path
$frontendDir = Join-Path $appDir 'frontend'
$backendDir = Join-Path $appDir 'backend'
$frontendHost = if ($env:FACEAI_FRONTEND_HOST) { $env:FACEAI_FRONTEND_HOST } else { '0.0.0.0' }
$backendHost = if ($env:FACEAI_BACKEND_HOST) { $env:FACEAI_BACKEND_HOST } else { '0.0.0.0' }
$backendPort = if ($env:FACEAI_BACKEND_PORT) { $env:FACEAI_BACKEND_PORT } else { '8000' }
$viteBin = Join-Path $frontendDir 'node_modules\vite\bin\vite.js'

$pythonCandidates = @(
  $env:FACEAI_PYTHON,
  $env:PYTHON_CMD,
  (Join-Path $backendDir '.venv\Scripts\python.exe'),
  (Join-Path $appDir '.venv\Scripts\python.exe'),
  (Join-Path $repoRoot '.venv\Scripts\python.exe'),
  (Join-Path $backendDir '.venv\bin\python'),
  (Join-Path $appDir '.venv\bin\python'),
  (Join-Path $repoRoot '.venv\bin\python'),
  'python',
  'python3',
  'py -3'
) | Where-Object { $_ }

function Find-WorkingPython {
  foreach ($candidate in $pythonCandidates) {
    try {
      if ($candidate -match '\s' -and -not (Test-Path $candidate)) {
        Invoke-Expression "$candidate --version" *> $null
      } else {
        & $candidate --version *> $null
      }

      return $candidate
    } catch {
      continue
    }
  }

  return $null
}

function Ensure-BackendPackages([string]$pythonCommand) {
  try {
    if ($pythonCommand -match '\s' -and -not (Test-Path $pythonCommand)) {
      Invoke-Expression "$pythonCommand -c ""import fastapi, uvicorn""" *> $null
    } else {
      & $pythonCommand -c "import fastapi, uvicorn" *> $null
    }
  } catch {
    $pythonHint = if ($IsWindows) { 'faceai/backend/.venv/Scripts/python' } else { 'faceai/backend/.venv/bin/python' }
    throw "Backend Python packages are not installed yet. Run `"$pythonHint -m pip install -r faceai/backend/requirements.txt`" and then try again."
  }
}

if (-not (Test-Path $viteBin)) {
  throw 'Frontend dependencies are missing. Run `npm install` from the repo root (or from `faceai/`) and then try again.'
}

$pythonCommand = Find-WorkingPython
if (-not $pythonCommand) {
  throw 'Python 3 was not found on this machine. Install Python 3.11+ and create `faceai/backend/.venv`, or set FACEAI_PYTHON to a working Python executable path.'
}

Ensure-BackendPackages $pythonCommand

Write-Host 'Starting FaceAI dev servers...'
Write-Host 'Frontend: http://localhost:5173'
Write-Host "Backend:  http://localhost:$backendPort"

$frontendProcess = Start-Process -FilePath 'cmd.exe' `
  -ArgumentList '/d', '/s', '/c', "npm.cmd --prefix frontend run dev -- --host $frontendHost" `
  -WorkingDirectory $appDir `
  -NoNewWindow `
  -PassThru

$backendArgs = @(
  '-m',
  'uvicorn',
  'app.main:app',
  '--reload',
  '--reload-exclude',
  'model_cache',
  '--reload-exclude',
  'backend/model_cache',
  '--host',
  $backendHost,
  '--port',
  $backendPort,
  '--app-dir',
  'backend'
)

if ($pythonCommand -match '\s' -and -not (Test-Path $pythonCommand)) {
  $backendProcess = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList '-NoProfile', '-Command', "$pythonCommand $($backendArgs -join ' ')" `
    -WorkingDirectory $appDir `
    -NoNewWindow `
    -PassThru
} else {
  $backendProcess = Start-Process -FilePath $pythonCommand `
    -ArgumentList $backendArgs `
    -WorkingDirectory $appDir `
    -NoNewWindow `
    -PassThru
}

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
