import { NextResponse, NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File;

  if (!file) {
    return NextResponse.json({ error: "No file" }, { status: 400 });
  }

  // move to fastapi
  const backendRes = await fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData,
  });

  const data = await backendRes.json();
  return NextResponse.json(data);
}
