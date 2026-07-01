// Simplified thread pool that uses the original process-based approach
// but with proper sequential batch processing

class SimpleThreadPool {
  constructor() {
    this.activeProcesses = new Map(); // track active child processes by runId
    this.pendingTasks = [];
  }

  // This simplified version just tracks active processes
  // The actual processing is handled by the original spawn logic in upload/route.ts
  
  registerProcess(runId, childProcess) {
    this.activeProcesses.set(runId, childProcess);
    return true;
  }

  unregisterProcess(runId) {
    this.activeProcesses.delete(runId);
    return true;
  }

  // For backward compatibility with cancel logic
  cancelTask(runId) {
    const process = this.activeProcesses.get(runId);
    if (process) {
      try {
        process.kill('SIGTERM');
        this.activeProcesses.delete(runId);
        return true;
      } catch (e) {
        return false;
      }
    }
    return false;
  }

  // Check if a run was cancelled
  wasCancelled(runId) {
    return false; // Not tracking cancellations in this simple version
  }

  // Clear cancelled state
  clearCancelled(runId) {
    // No-op for this simple version
  }
}

module.exports = new SimpleThreadPool();