import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import crypto from "crypto";
import { spawn } from "child_process";
import { eventBus } from "@/lib/events";

export async function POST(req: NextRequest) {
  try {
    const data = await req.formData();
    const file = data.get("file");

    if (!file || !(file instanceof File)) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Convert image to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const runId = crypto.randomUUID();
    const backendDir = path.join(process.cwd(), "backend");
    const uploadsDir = path.join(backendDir, "uploads");

    await fs.mkdir(uploadsDir, { recursive: true });

    // Save under the original file name for backward-compatibility
    const originalUploadPath = path.join(uploadsDir, file.name);
    await fs.writeFile(originalUploadPath, buffer);

    // Save with the run ID for the processing pipeline
    const safeExt = path.extname(file.name || "").toLowerCase() || ".jpg";
    const uploadPath = path.join(uploadsDir, `input_${runId}${safeExt}`);
    await fs.writeFile(uploadPath, buffer);

    // Notify listeners that a new upload has started
    eventBus.emit({ type: "start", runId, timestamp: Date.now() });

    const resultPath = path.join(uploadsDir, `result_${runId}.json`);
    const scriptPath = path.join(backendDir, "run_ecg_analysis.py");
    const venvPythonPath = path.join(process.cwd(), "..", "env", "Scripts", "python.exe");
    const args = [scriptPath, "--input", uploadPath, "--run-id", runId, "--result-path", resultPath];

    const trySpawnPython = (pythonCmd: string) =>
      new Promise<{ code: number; stderrTail: string }>((resolve) => {
        let stdoutTail = "";
        let stderrTail = "";
        let resolved = false;

        const p = spawn(pythonCmd, args, {
          cwd: process.cwd(),
          shell: false,
        });

        p.on("error", (err) => {
          if (!resolved) {
            resolved = true;
            resolve({ code: 1, stderrTail: String(err) });
          }
        });

        p.stdout.on("data", (chunk) => {
          const text = chunk.toString("utf8");
          stdoutTail += text;
          
          // Parse progress messages
          const lines = text.split("\n");
          for (const line of lines) {
            if (line.startsWith("PROGRESS:")) {
              try {
                const progressData = JSON.parse(line.substring(9));
                eventBus.emit({ ...progressData, runId });
              } catch (e) {
                console.error("Failed to parse progress JSON", e);
              }
            }
          }

          if (stdoutTail.length > 20000) stdoutTail = stdoutTail.slice(-20000);
        });
        p.stderr.on("data", (chunk) => {
          stderrTail += chunk.toString("utf8");
          if (stderrTail.length > 20000) stderrTail = stderrTail.slice(-20000);
        });

        p.on("close", (code) => {
          if (!resolved) {
            resolved = true;
            resolve({ code: code ?? 1, stderrTail: stderrTail || stdoutTail });
          }
        });
      });

    const venvAttempt = await trySpawnPython(venvPythonPath);
    const { code, stderrTail } =
      venvAttempt.code === 0 
        ? venvAttempt 
        : await (async () => {
            const pythonAttempt = await trySpawnPython("python");
            return pythonAttempt.code === 0 ? pythonAttempt : await trySpawnPython("py");
          })();

    if (code !== 0) {
      eventBus.emit({ type: "error", runId, error: "ECG analysis failed" });
      return NextResponse.json({
        success: false,
        error: "ECG analysis failed",
        details: stderrTail,
      }, { status: 500 });
    }

    const resultRaw = await fs.readFile(resultPath, "utf-8");
    const result = JSON.parse(resultRaw);

    eventBus.emit({ type: "complete", runId, result });

    return NextResponse.json({
      message: "File uploaded successfully",
      filename: file.name,
      ...result
    });
  } catch (error: any) {
    console.error(error);
    return NextResponse.json({ error: "Upload failed", details: error.message }, { status: 500 });
  }
}
