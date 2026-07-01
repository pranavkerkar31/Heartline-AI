const { parentPort, workerData } = require('worker_threads');
const { spawn } = require('child_process');
const path = require('path');

async function trySpawnPython(pythonCmd, args) {
  return new Promise((resolve) => {
    let stdoutTail = "";
    let stderrTail = "";
    let resolved = false;

    const child = spawn(pythonCmd, args, {
      cwd: process.cwd(),
      shell: false,
    });

    child.on("error", (err) => {
      if (!resolved) {
        resolved = true;
        resolve({ code: 1, stderrTail: String(err) });
      }
    });

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      stdoutTail += text;
      if (stdoutTail.length > 20000) stdoutTail = stdoutTail.slice(-20000);
    });

    child.stderr.on("data", (chunk) => {
      stderrTail += chunk.toString("utf8");
      if (stderrTail.length > 20000) stderrTail = stderrTail.slice(-20000);
    });

    child.on("close", (code) => {
      if (!resolved) {
        resolved = true;
        resolve({ code: code ?? 1, stderrTail: stderrTail || stdoutTail });
      }
    });
  });
}

async function executeTask() {
  try {
    const task = workerData;
    
    const venvPythonPath = path.join(process.cwd(), "..", "env", "Scripts", "python.exe");
    const venvAttempt = await trySpawnPython(venvPythonPath, task.args);
    const { code, stderrTail } = 
      venvAttempt.code === 0 
        ? venvAttempt
        : await (async () => {
            const pythonAttempt = await trySpawnPython("python", task.args);
            if (pythonAttempt.code === 0) return pythonAttempt;
            return await trySpawnPython("py", task.args);
          })();
    
    parentPort.postMessage({
      runId: task.runId,
      code,
      stderrTail
    });
  } catch (error) {
    parentPort.postMessage({
      runId: workerData.runId,
      code: 1,
      stderrTail: error instanceof Error ? error.message : "Unknown error"
    });
  }
}

executeTask();