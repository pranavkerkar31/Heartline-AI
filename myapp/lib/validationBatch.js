// JavaScript version of validationBatch for immediate use
exports.VALIDATION_CATEGORIES = ["HB", "MI", "Normal", "PMI"];

class ValidationBatchManager {
  constructor() {
    this.batch = null;
    this.currentRecordClaims = new Set();
  }

  configure(category, start, end) {
    this.batch = { category, start, end, current: start, activeRunId: null, complete: false };
    this.currentRecordClaims.clear();
    return this.snapshot();
  }

  snapshot() {
    return this.batch ? { ...this.batch } : null;
  }

  claim(runId) {
    if (!this.batch) throw new Error("Configure a validation range in Live Monitor first.");
    if (this.batch.complete) throw new Error("The configured validation range is already complete.");
    
    // For sequential processing: assign current record and immediately advance
    const recordNumber = this.batch.current;
    
    // Advance to next record immediately
    if (this.batch.current < this.batch.end) {
      this.batch.current += 1;
    } else {
      this.batch.complete = true;
    }
    
    // Track which runIds are working on records (for cleanup)
    this.currentRecordClaims.add(runId);
    
    return { category: this.batch.category, recordNumber };
  }

  succeed(runId) {
    if (!this.batch) return this.snapshot();
    
    // Remove this runId from current record claims
    this.currentRecordClaims.delete(runId);
    
    // No need to advance here - we advance immediately in claim() for sequential processing
    return this.snapshot();
  }

  release(runId) {
    if (!this.batch) return this.snapshot();
    
    if (this.batch?.activeRunId === runId) {
      this.batch.activeRunId = null;
    }
    this.currentRecordClaims.delete(runId);
    
    // Note: With sequential assignment, if a claim is released (failed/cancelled),
    // the record number has already been advanced and won't be reused.
    // This means skipped record numbers can occur if uploads fail.
    
    return this.snapshot();
  }
}

const globalForValidationBatch = globalThis;
globalForValidationBatch.validationBatchManager = globalForValidationBatch.validationBatchManager ?? new ValidationBatchManager();

exports.validationBatchManager = globalForValidationBatch.validationBatchManager;