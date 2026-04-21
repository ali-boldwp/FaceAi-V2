import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(appDir, "..");
const frontendDir = path.join(appDir, "frontend");
const backendDir = path.join(appDir, "backend");
const viteCli = path.join(frontendDir, "node_modules", "vite", "bin", "vite.js");
const backendRequirements = path.join(backendDir, "requirements.txt");
const backendHost = process.env.FACEAI_BACKEND_HOST ?? "0.0.0.0";
const backendPort = process.env.FACEAI_BACKEND_PORT ?? "8000";
const frontendHost = process.env.FACEAI_FRONTEND_HOST ?? "0.0.0.0";

const pythonCandidates = [
  process.env.FACEAI_PYTHON ? { command: process.env.FACEAI_PYTHON, args: [] } : null,
  process.env.PYTHON_CMD ? { command: process.env.PYTHON_CMD, args: [] } : null,
  { command: path.join(backendDir, ".venv", "Scripts", "python.exe"), args: [] },
  { command: path.join(appDir, ".venv", "Scripts", "python.exe"), args: [] },
  { command: path.join(repoRoot, ".venv", "Scripts", "python.exe"), args: [] },
  { command: path.join(backendDir, ".venv", "bin", "python"), args: [] },
  { command: path.join(appDir, ".venv", "bin", "python"), args: [] },
  { command: path.join(repoRoot, ".venv", "bin", "python"), args: [] },
  { command: "python", args: [] },
  { command: "python3", args: [] },
  { command: "py", args: ["-3"] },
].filter(Boolean);

const children = [];
let shuttingDown = false;

function formatCommand(command, args) {
  return [command, ...args].join(" ");
}

function fail(message) {
  console.error(`\n${message}\n`);
  process.exit(1);
}

function findWorkingPython() {
  for (const candidate of pythonCandidates) {
    const result = spawnSync(candidate.command, [...candidate.args, "--version"], {
      encoding: "utf8",
      stdio: "pipe",
      windowsHide: true,
    });

    if (result.error || result.status !== 0) {
      continue;
    }

    return candidate;
  }

  return null;
}

function ensureBackendPackages(python) {
  const result = spawnSync(
    python.command,
    [...python.args, "-c", "import fastapi, uvicorn"],
    {
      cwd: appDir,
      encoding: "utf8",
      stdio: "pipe",
      windowsHide: true,
    },
  );

  if (result.error || result.status !== 0) {
    const pythonHint =
      process.platform === "win32"
        ? "faceai/backend/.venv/Scripts/python"
        : "faceai/backend/.venv/bin/python";

    fail(
      [
        "Backend Python packages are not installed yet.",
        `Run \`${pythonHint} -m pip install -r faceai/backend/requirements.txt\` and then try again.`,
      ].join("\n"),
    );
  }
}

function terminateChild(child) {
  if (!child || child.exitCode !== null) {
    return;
  }

  if (process.platform === "win32") {
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
      stdio: "ignore",
      windowsHide: true,
    });
    return;
  }

  child.kill("SIGTERM");
}

function shutdown(exitCode = 0) {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;
  for (const child of children) {
    terminateChild(child);
  }

  setTimeout(() => {
    process.exit(exitCode);
  }, 200);
}

function launch(name, command, args, options) {
  console.log(`[${name}] ${formatCommand(command, args)}`);

  const child = spawn(command, args, {
    stdio: "inherit",
    windowsHide: false,
    ...options,
  });

  children.push(child);

  child.on("error", (error) => {
    console.error(`\n${name} failed to start: ${error.message}`);
    shutdown(1);
  });

  child.on("exit", (code, signal) => {
    if (shuttingDown) {
      return;
    }

    const detail = signal ? `signal ${signal}` : `code ${code ?? 0}`;
    console.error(`\n${name} exited with ${detail}. Stopping the other process.`);
    shutdown(code ?? 1);
  });

  return child;
}

if (!existsSync(viteCli)) {
  fail(
    [
      "Frontend dependencies are missing.",
      "Run `npm install` from the repo root (or from `faceai/`) and then try again.",
    ].join("\n"),
  );
}

if (!existsSync(backendRequirements)) {
  fail(`Could not find backend requirements at ${backendRequirements}.`);
}

const python = findWorkingPython();

if (!python) {
  fail(
    [
      "Python 3 was not found on this machine.",
      "Install Python 3.11+ and create `faceai/backend/.venv`, then install the backend requirements.",
      "You can also set `FACEAI_PYTHON` to a working Python executable path.",
    ].join("\n"),
  );
}

ensureBackendPackages(python);

console.log("Starting FaceAI dev servers...");
console.log(`Frontend: http://localhost:5173`);
console.log(`Backend:  http://localhost:${backendPort}`);

launch("frontend", process.execPath, [viteCli, "--host", frontendHost], {
  cwd: frontendDir,
});

launch(
  "backend",
  python.command,
  [
    ...python.args,
    "-m",
    "uvicorn",
    "app.main:app",
    "--reload",
    "--reload-exclude",
    "model_cache",
    "--reload-exclude",
    "backend/model_cache",
    "--host",
    backendHost,
    "--port",
    backendPort,
    "--app-dir",
    "backend",
  ],
  {
    cwd: appDir,
  },
);

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
