"use client";

import React, { useEffect, useRef, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { readNpzEcg, renderEcgToCanvas } from "@/lib/ecg-paper-renderer";

interface ECGReconstructionViewerProps {
  runId: string;
  className?: string;
  onPreviewReady?: (dataUrl: string) => void;
}

export default function ECGReconstructionViewer({
  runId,
  className = "",
  onPreviewReady,
}: ECGReconstructionViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewReadyRef = useRef(onPreviewReady);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    previewReadyRef.current = onPreviewReady;
  }, [onPreviewReady]);

  useEffect(() => {
    let cancelled = false;

    async function renderReconstruction() {
      const canvas = canvasRef.current;
      if (!canvas) return;

      try {
        setStatus("loading");
        setError("");

        const ecg = await readNpzEcg(`/api/ecg-reconstruction/${runId}`);
        const result = renderEcgToCanvas(ecg, canvas, {
          dpi: 140,
          showHeader: true,
          metadata: { id: runId.slice(0, 8) },
        });

        if (cancelled) return;

        setStatus("ready");
        previewReadyRef.current?.(result.canvas.toDataURL("image/png"));
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Unable to render ECG reconstruction");
      }
    }

    renderReconstruction();

    return () => {
      cancelled = true;
    };
  }, [runId]);

  return (
    <div className={`relative w-full h-full bg-white ${className}`}>
      <canvas
        ref={canvasRef}
        aria-label="Rendered ECG reconstruction"
        className={`w-full h-full object-contain transition-opacity duration-300 ${
          status === "ready" ? "opacity-100" : "opacity-20"
        }`}
      />

      {status === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-teal-50/80">
          <Loader2 className="w-8 h-8 text-teal-500 animate-spin" />
        </div>
      )}

      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-red-50 p-3 text-center">
          <AlertCircle className="w-7 h-7 text-red-400" />
          <span className="text-[11px] font-semibold text-red-600 line-clamp-3">{error}</span>
        </div>
      )}
    </div>
  );
}
