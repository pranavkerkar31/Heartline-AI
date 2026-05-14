import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import { createReadStream } from "fs";

export async function GET(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    // Await params as per Next.js 15+ standards if needed, 
    // but here we just access them. 
    // Wait, in Next.js 15 params is a promise. 
    const resolvedParams = await (params as any);
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

    return new NextResponse(stream as any, {
      headers: {
        "Content-Type": contentType,
        "Content-Length": stats.size.toString(),
      },
    });
  } catch (error) {
    return new NextResponse("Not Found", { status: 404 });
  }
}
