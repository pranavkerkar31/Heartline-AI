import type { ChildProcess } from "child_process";

class RunRegistry {
  private processes = new Map<string, ChildProcess>();
  private cancelled = new Set<string>();

  register(runId: string, process: ChildProcess) {
    this.processes.set(runId, process);
    this.cancelled.delete(runId);
  }

  unregister(runId: string) {
    this.processes.delete(runId);
  }

  cancel(runId: string) {
    const proc = this.processes.get(runId);
    if (!proc) return false;

    this.cancelled.add(runId);
    // Use SIGKILL to forcefully terminate the process and its children
    // On Windows, this is 'SIGKILL' which is the same as force kill
    // On Unix, we use SIGKILL (-9) which cannot be caught or ignored
    if (process.platform === 'win32') {
      proc.kill('SIGKILL');
    } else {
      // On Unix, try to kill the entire process group
      try {
        process.kill(-proc.pid, 'SIGKILL');
      } catch (e) {
        // If killing the process group fails, fall back to killing just the process
        proc.kill('SIGKILL');
      }
    }
    this.processes.delete(runId);
    return true;
  }

  wasCancelled(runId: string) {
    return this.cancelled.has(runId);
  }

  clearCancelled(runId: string) {
    this.cancelled.delete(runId);
  }
}

const globalForRunRegistry = globalThis as unknown as {
  runRegistry: RunRegistry | undefined;
};

export const runRegistry = globalForRunRegistry.runRegistry ?? new RunRegistry();

if (process.env.NODE_ENV !== "production") {
  globalForRunRegistry.runRegistry = runRegistry;
}
