import { NextRequest, NextResponse } from "next/server";
import { eventBus } from "@/lib/events";
import {
  VALIDATION_CATEGORIES,
  validationBatchManager,
  type ValidationCategory,
} from "@/lib/validationBatch";

export async function GET() {
  return NextResponse.json({ batch: validationBatchManager.snapshot() });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const category = String(body?.category ?? "") as ValidationCategory;
  const start = Number(body?.start);
  const end = Number(body?.end);

  if (!VALIDATION_CATEGORIES.includes(category)) {
    return NextResponse.json({ error: "Invalid ECG category" }, { status: 400 });
  }
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) {
    return NextResponse.json({ error: "Enter a valid inclusive record range" }, { status: 400 });
  }

  if (validationBatchManager.snapshot()?.activeRunId) {
    return NextResponse.json({ error: "Cancel the active phone scan before changing the range." }, { status: 409 });
  }

  const batch = validationBatchManager.configure(category, start, end);
  eventBus.emit({ type: "batch", batch, timestamp: Date.now() });
  return NextResponse.json({ batch });
}
