import { NextRequest, NextResponse } from "next/server";
import { eventBus } from "@/lib/events";
import { runRegistry } from "@/lib/runRegistry";
import { validationBatchManager } from "@/lib/validationBatch";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const runId = String(body?.runId ?? "").trim();
    const category = body?.category ?? null;
    const recordNumber = body?.recordNumber ?? null;

    if (!runId) {
      return NextResponse.json({ success: false, error: "Missing runId" }, { status: 400 });
    }

    const cancelled = runRegistry.cancel(runId);
    
    // Always release from batch if it was assigned, regardless of whether process was found
    const batch = validationBatchManager.release(runId);
    
    // Emit cancelled event only if we actually cancelled a running process
    if (cancelled) {
      eventBus.emit({ type: "cancelled", runId, category, recordNumber, timestamp: Date.now() });
    }

    eventBus.emit({ type: "batch", batch, timestamp: Date.now() });

    return NextResponse.json({ success: cancelled, runId });
  } catch {
    return NextResponse.json({ success: false, error: "Failed to cancel run" }, { status: 500 });
  }
}

