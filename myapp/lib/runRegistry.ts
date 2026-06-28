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
    proc.kill();
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
