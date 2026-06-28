import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import crypto from "crypto";
import { spawn } from "child_process";
import { eventBus } from "@/lib/events";
import { runRegistry } from "@/lib/runRegistry";
import { validationBatchManager } from "@/lib/validationBatch";

type UploadLike = Blob & {
  arrayBuffer(): Promise<ArrayBuffer>;
  name?: string;
};

export async function POST(req: NextRequest) {
  let liveBatchRunId: string | null = null;

  try {
    const data = await req.formData();

    const entry = data.get("file") ?? data.get("image");
    const maybeFile = entry as UploadLike | null;
    const file =
      maybeFile && typeof maybeFile.arrayBuffer === "function" && typeof maybeFile.name === "string"
        ? (maybeFile as File)
        : null;

    if (!file) {
      return NextResponse.json({
        success: false,
        error: "No image uploaded",
      });
    }

    let category = String(data.get("category") ?? "").trim() || null;
    let recordNumber: string | number | null = String(data.get("recordNumber") ?? "").trim() || null;

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const runId = crypto.randomUUID();
    const backendDir = path.join(process.cwd(), "backend");
    const uploadsDir = path.join(backendDir, "uploads");
    const usesLiveBatch = !category || !recordNumber;

    if (usesLiveBatch) {
      try {
        const target = validationBatchManager.claim(runId);
        liveBatchRunId = runId;
        category = target.category;
        recordNumber = target.recordNumber;
      } catch (error) {
        return NextResponse.json(
          { success: false, error: error instanceof Error ? error.message : "Validation batch is unavailable" },
          { status: 409 },
        );
      }
    }

    await fs.mkdir(uploadsDir, { recursive: true });

    const safeExt = path.extname(file.name || "").toLowerCase() || ".jpg";
    const uploadPath = path.join(uploadsDir, `input_${runId}${safeExt}`);

    await fs.writeFile(uploadPath, buffer);

    eventBus.emit({
      type: "start",
      runId,
      category,
      recordNumber,
      timestamp: Date.now(),
    });

    const resultPath = path.join(uploadsDir, `result_${runId}.json`);
    const scriptPath = path.join(backendDir, "run_ecg_analysis.py");
    const venvPythonPath = path.join(process.cwd(), "..", "env", "Scripts", "python.exe");
    const args = [scriptPath, "--input", uploadPath, "--run-id", runId, "--result-path", resultPath];

    if (category) args.push("--category", category);
    if (recordNumber) args.push("--record-number", String(recordNumber));

    const trySpawnPython = (pythonCmd: string) =>
      new Promise<{ code: number; stderrTail: string }>((resolve) => {
        let stdoutTail = "";
        let stderrTail = "";
        let resolved = false;

        const child = spawn(pythonCmd, args, {
          cwd: process.cwd(),
          shell: false,
        });

        runRegistry.register(runId, child);

        child.on("error", (err) => {
          runRegistry.unregister(runId);
          if (!resolved) {
            resolved = true;
            resolve({ code: 1, stderrTail: String(err) });
          }
        });

        child.stdout.on("data", (chunk) => {
          const text = chunk.toString("utf8");
          stdoutTail += text;

          const lines = text.split("\n");
          for (const line of lines) {
            if (line.startsWith("PROGRESS:")) {
              try {
                const progressData = JSON.parse(line.substring(9));
                eventBus.emit({ ...progressData, runId, category, recordNumber });
              } catch (e) {
                console.error("Failed to parse progress JSON", e);
              }
            }
          }

          if (stdoutTail.length > 20000) stdoutTail = stdoutTail.slice(-20000);
        });

        child.stderr.on("data", (chunk) => {
          stderrTail += chunk.toString("utf8");
          if (stderrTail.length > 20000) stderrTail = stderrTail.slice(-20000);
        });

        child.on("close", (code) => {
          runRegistry.unregister(runId);
          if (!resolved) {
            resolved = true;
            resolve({ code: code ?? 1, stderrTail: stderrTail || stdoutTail });
          }
        });
      });

    const venvAttempt = await trySpawnPython(venvPythonPath);
    const { code, stderrTail } =
      venvAttempt.code === 0 || runRegistry.wasCancelled(runId)
        ? venvAttempt
        : await (async () => {
            const pythonAttempt = await trySpawnPython("python");
            if (pythonAttempt.code === 0 || runRegistry.wasCancelled(runId)) return pythonAttempt;
            return await trySpawnPython("py");
          })();

    if (code !== 0) {
      const cancelled = runRegistry.wasCancelled(runId);
      runRegistry.clearCancelled(runId);
      const batch = usesLiveBatch ? validationBatchManager.release(runId) : validationBatchManager.snapshot();
      const errorMessage = cancelled ? "Processing cancelled" : "ECG analysis failed";
      eventBus.emit({ type: cancelled ? "cancelled" : "error", runId, category, recordNumber, error: errorMessage });
      if (usesLiveBatch) eventBus.emit({ type: "batch", batch, timestamp: Date.now() });
      return NextResponse.json({
        success: false,
        cancelled,
        runId,
        error: errorMessage,
        details: stderrTail,
      });
    }

    const resultRaw = await fs.readFile(resultPath, "utf-8");
    const result = JSON.parse(resultRaw);

    if (!result.success || !result.validation) {
      const batch = usesLiveBatch ? validationBatchManager.release(runId) : validationBatchManager.snapshot();
      const errorMessage = result.error || "Validation did not complete";
      eventBus.emit({ type: "error", runId, category, recordNumber, error: errorMessage });
      if (usesLiveBatch) eventBus.emit({ type: "batch", batch, timestamp: Date.now() });
      return NextResponse.json({ ...result, runId, error: errorMessage }, { status: 500 });
    }

    const batch = usesLiveBatch ? validationBatchManager.succeed(runId) : validationBatchManager.snapshot();
    liveBatchRunId = null;
    eventBus.emit({ type: "complete", runId, category, recordNumber, result, batch });
    if (usesLiveBatch) eventBus.emit({ type: "batch", batch, timestamp: Date.now() });
    return NextResponse.json(result);
  } catch (error) {
    if (liveBatchRunId) {
      const batch = validationBatchManager.release(liveBatchRunId);
      eventBus.emit({ type: "batch", batch, timestamp: Date.now() });
    }
    console.log(error);
    return NextResponse.json({
      success: false,
      error: "Upload failed",
    });
  }
}
