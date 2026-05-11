import { writeFile } from "fs/promises";
import path from "path";

export async function POST(req: Request) {
  try {
    const data = await req.formData();
    const file = data.get("file");

    if (!file || !(file instanceof File)) {
      return Response.json({ error: "No file uploaded" }, { status: 400 });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Save inside uploads folder
    const uploadPath = path.join(
      process.cwd(),
      "backend",
      "uploads",
      file.name,
    );

    await writeFile(uploadPath, buffer);

    return Response.json({
      message: "File uploaded successfully",
      filename: file.name,
    });
  } catch (error) {
    console.error(error);

    return Response.json({ error: "Upload failed" }, { status: 500 });
  }
}
