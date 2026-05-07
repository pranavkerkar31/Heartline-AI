import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";

export async function POST(req: NextRequest) {
  try {
    const data = await req.formData();

    const file = data.get("image") as File;

    if (!file) {
      return NextResponse.json({
        success: false,
        error: "No image uploaded",
      });
    }

    // Convert image to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Save image path
    const uploadPath = path.join(process.cwd(), "./backend/uploads/input.jpg");

    // Save image
    console.log("File received:", file.name);

    console.log("Current working directory:", process.cwd());

    console.log("Upload path:", uploadPath);
    await fs.writeFile(uploadPath, buffer);

    return NextResponse.json({
      success: true,
      message: "Image uploaded successfully",
    });
  } catch (error) {
    console.log(error);

    return NextResponse.json({
      success: false,
      error: "Upload failed",
    });
  }
}
