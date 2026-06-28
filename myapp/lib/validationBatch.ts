export const VALIDATION_CATEGORIES = ["HB", "MI", "Normal", "PMI"] as const;

export type ValidationCategory = (typeof VALIDATION_CATEGORIES)[number];

export interface ValidationBatchSnapshot {
  category: ValidationCategory;
  start: number;
  end: number;
  current: number;
  activeRunId: string | null;
  complete: boolean;
}

class ValidationBatchManager {
  private batch: ValidationBatchSnapshot | null = null;

  configure(category: ValidationCategory, start: number, end: number) {
    this.batch = { category, start, end, current: start, activeRunId: null, complete: false };
    return this.snapshot();
  }

  snapshot() {
    return this.batch ? { ...this.batch } : null;
  }

  claim(runId: string) {
    if (!this.batch) throw new Error("Configure a validation range in Live Monitor first.");
    if (this.batch.complete) throw new Error("The configured validation range is already complete.");
    if (this.batch.activeRunId) {
      throw new Error(`Another scan is already processing for ${this.batch.category} ${this.batch.current}.`);
    }
    this.batch.activeRunId = runId;
    return { category: this.batch.category, recordNumber: this.batch.current };
  }

  succeed(runId: string) {
    if (!this.batch || this.batch.activeRunId !== runId) return this.snapshot();
    if (this.batch.current >= this.batch.end) this.batch.complete = true;
    else this.batch.current += 1;
    this.batch.activeRunId = null;
    return this.snapshot();
  }

  release(runId: string) {
    if (this.batch?.activeRunId === runId) this.batch.activeRunId = null;
    return this.snapshot();
  }
}

const globalForValidationBatch = globalThis as unknown as {
  validationBatchManager: ValidationBatchManager | undefined;
};

export const validationBatchManager =
  globalForValidationBatch.validationBatchManager ?? new ValidationBatchManager();
globalForValidationBatch.validationBatchManager = validationBatchManager;
