import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import { createReadStream } from "fs";
import { Readable } from "stream";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const filePath = path.join(process.cwd(), "backend", "uploads", ...resolvedParams.path);

    // Basic security: ensure the path is inside backend/uploads
    const absoluteUploadsDir = path.join(process.cwd(), "backend", "uploads");
    if (!filePath.startsWith(absoluteUploadsDir)) {
      return new NextResponse("Forbidden", { status: 403 });
    }

    const stats = await fs.stat(filePath);
    if (!stats.isFile()) {
      return new NextResponse("Not Found", { status: 404 });
    }

    const stream = createReadStream(filePath);
    
    // Determine content type
    const ext = path.extname(filePath).toLowerCase();
    const contentType = {
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".svg": "image/svg+xml",
      ".json": "application/json",
      ".npz": "application/octet-stream",
    }[ext] || "application/octet-stream";

    return new NextResponse(Readable.toWeb(stream) as ReadableStream, {
      headers: {
        "Content-Type": contentType,
        "Content-Length": stats.size.toString(),
      },
    });
  } catch {
    return new NextResponse("Not Found", { status: 404 });
  }
}
