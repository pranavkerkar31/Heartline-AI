import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import { Readable } from "stream";

function getNpzPath(runId: string) {
  return path.join(
    process.cwd(),
    "backend",
    "uploads",
    "digitized",
    runId,
    "ecg_signals_mv.npz"
  );
}

function isValidRunId(runId: string) {
  return /^[a-zA-Z0-9_-]+$/.test(runId);
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ runId: string }> | { runId: string } }
) {
  try {
    const { runId } = await params;

    if (!isValidRunId(runId)) {
      return new NextResponse("Invalid run ID", { status: 400 });
    }

    const uploadsDir = path.join(process.cwd(), "backend", "uploads", "digitized");
    const filePath = getNpzPath(runId);
    const relative = path.relative(uploadsDir, filePath);

    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      return new NextResponse("Forbidden", { status: 403 });
    }

    const stats = await fs.stat(filePath);
    if (!stats.isFile()) {
      return new NextResponse("Not Found", { status: 404 });
    }

    return new NextResponse(Readable.toWeb(createReadStream(filePath)) as ReadableStream, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `inline; filename="ecg_signals_mv_${runId}.npz"`,
        "Content-Length": stats.size.toString(),
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new NextResponse("Not Found", { status: 404 });
  }
}

export async function HEAD(
  _req: NextRequest,
  { params }: { params: Promise<{ runId: string }> | { runId: string } }
) {
  try {
    const { runId } = await params;

    if (!isValidRunId(runId)) {
      return new NextResponse(null, { status: 400 });
    }

    const stats = await fs.stat(getNpzPath(runId));
    return new NextResponse(null, {
      status: stats.isFile() ? 200 : 404,
      headers: {
        "Content-Length": stats.size.toString(),
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new NextResponse(null, { status: 404 });
  }
}
