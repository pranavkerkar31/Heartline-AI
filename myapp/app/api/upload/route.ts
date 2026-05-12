import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import crypto from "crypto";
import { spawn } from "child_process";

export async function POST(req: NextRequest) {
  try {
    const data = await req.formData();

    const entry = data.get("file") ?? data.get("image");
    // `File` might not always be a reliable `instanceof` check in server runtimes;
    // instead, accept anything that looks like a File (has `arrayBuffer()`).
    const file =
      entry &&
      typeof (entry as any).arrayBuffer === "function" &&
      typeof (entry as any).name === "string"
        ? (entry as File)
        : null;

    if (!file) {
      return NextResponse.json({
        success: false,
        error: "No image uploaded",
      });
    }

    // Convert image to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const runId = crypto.randomUUID();
    const backendDir = path.join(process.cwd(), "backend");
    const uploadsDir = path.join(backendDir, "uploads");

    await fs.mkdir(uploadsDir, { recursive: true });

    // Save uploaded image to a temp input path (Python will stage/copy it for YOLO).
    const safeExt = path.extname(file.name || "").toLowerCase() || ".jpg";
    const uploadPath = path.join(uploadsDir, `input_${runId}${safeExt}`);

    // Save image
    await fs.writeFile(uploadPath, buffer);

    const resultPath = path.join(uploadsDir, `result_${runId}.json`);
    const scriptPath = path.join(backendDir, "run_ecg_analysis.py");

    // Path to your virtual environment's python executable
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
          stdoutTail += chunk.toString("utf8");
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

    // 1. Try Virtual Env first, then fallback to 'python' or 'py'
    const venvAttempt = await trySpawnPython(venvPythonPath);
    const { code, stderrTail } =
      venvAttempt.code === 0 
        ? venvAttempt 
        : await (async () => {
            const pythonAttempt = await trySpawnPython("python");
            return pythonAttempt.code === 0 ? pythonAttempt : await trySpawnPython("py");
          })();

    if (code !== 0) {
      return NextResponse.json({
        success: false,
        error: "ECG analysis failed",
        details: stderrTail,
      });
    }

    const resultRaw = await fs.readFile(resultPath, "utf-8");
    const result = JSON.parse(resultRaw);

    return NextResponse.json(result);
  } catch (error) {
    console.log(error);

    return NextResponse.json({
      success: false,
      error: "Upload failed",
    });
  }
}
