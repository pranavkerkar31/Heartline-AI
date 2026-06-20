import { NextRequest } from "next/server";
import { eventBus } from "@/lib/events";

export async function GET(req: NextRequest) {
  console.log("SSE: Client connecting...");
  
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();

      const unsubscribe = eventBus.subscribe((data) => {
        console.log("SSE: Broadcasting event", data.type, "to client");
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      });

      req.signal.onabort = () => {
        console.log("SSE: Client disconnected");
        unsubscribe();
        controller.close();
      };
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
