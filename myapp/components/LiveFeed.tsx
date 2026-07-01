"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Download,
  Image as ImageIcon,
  Layers,
  Loader2,
  Maximize2,
  Save,
  Scissors,
  Square,
  X,
} from "lucide-react";

interface ProcessStep {
  id: string;
  label: string;
  status: "waiting" | "processing" | "complete" | "error";
  image?: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface ExportRow {
  category: string;
  record_number: number;
  lead: string;
  your_pcc: number | null;
  your_rmse: number | null;
  your_snr: number | null;
}

interface ValidationResult {
  export_rows: ExportRow[];
}

interface PipelineResult {
  validation?: ValidationResult;
}

interface Job {
  runId: string;
  timestamp: number;
  category?: string | null;
  recordNumber?: string | number | null;
  steps: ProcessStep[];
  error?: string;
  cancelled?: boolean;
  result?: PipelineResult;
}

interface BatchState {
  category: string;
  start: number;
  end: number;
  current: number;
  activeRunId: string | null;
  complete: boolean;
}

const STEP_DEFS: Array<Pick<ProcessStep, "id" | "label" | "icon">> = [
  { id: "received", label: "Received Photo", icon: ImageIcon },
  { id: "orientation", label: "Orientation Fix", icon: CheckCircle2 },
  { id: "yolo", label: "YOLO Detection", icon: Maximize2 },
  { id: "crop", label: "ECG Cropping", icon: Scissors },
  { id: "enhanced", label: "AI Enhancement", icon: Layers },
  { id: "mask", label: "Segmentation", icon: Activity },
  { id: "digitized", label: "Digitizer Output", icon: CheckCircle2 },
  { id: "validation", label: "Validation", icon: CheckCircle2 },
];

const metric = (value: number | null) => (value == null ? "" : String(value));

function downloadText(contents: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export default function LiveFeed() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [batch, setBatch] = useState<BatchState | null>(null);
  const [category, setCategory] = useState("MI");
  const [rangeStart, setRangeStart] = useState("1");
  const [rangeEnd, setRangeEnd] = useState("20");
  const [batchError, setBatchError] = useState("");
  const [savingBatch, setSavingBatch] = useState(false);
  const [selectedImage, setSelectedImage] = useState<{ src: string; label: string } | null>(null);

  useEffect(() => {
    fetch("/api/validation-batch")
      .then((response) => response.json())
      .then(({ batch: savedBatch }) => {
        if (!savedBatch) return;
        setBatch(savedBatch);
        setCategory(savedBatch.category);
        setRangeStart(String(savedBatch.start));
        setRangeEnd(String(savedBatch.end));
      })
      .catch(() => setBatchError("Could not load the current phone validation range."));
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedImage(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    const eventSource = new EventSource("/api/events");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "batch") {
          setBatch(data.batch);
          return;
        }

        if (data.type === "start") {
          const newJob: Job = {
            runId: data.runId,
            timestamp: data.timestamp,
            category: data.category,
            recordNumber: data.recordNumber,
            steps: STEP_DEFS.map((step, index) => ({
              ...step,
              status: index === 0 ? "complete" : index === 1 ? "processing" : "waiting",
            })),
          };
          setJobs((previous) => [newJob, ...previous].slice(0, 20));
          setBatch((currentBatch) =>
            currentBatch ? { ...currentBatch, activeRunId: data.runId } : currentBatch,
          );
          return;
        }

        if (data.type === "progress") {
          setJobs((previous) =>
            previous.map((job) => {
              if (job.runId !== data.runId) return job;
              const currentIndex = job.steps.findIndex((step) => step.id === data.step);
              if (currentIndex < 0) return job;
              return {
                ...job,
                steps: job.steps.map((step, index) => {
                  if (index < currentIndex) return { ...step, status: "complete" as const };
                  if (index === currentIndex) {
                    return { ...step, status: "complete" as const, image: data.image };
                  }
                  if (index === currentIndex + 1) return { ...step, status: "processing" as const };
                  return step;
                }),
              };
            }),
          );
          return;
        }

        if (data.type === "complete") {
          setJobs((previous) =>
            previous.map((job) =>
              job.runId === data.runId
                ? {
                    ...job,
                    category: data.category ?? job.category,
                    recordNumber: data.recordNumber ?? job.recordNumber,
                    result: data.result,
                    steps: job.steps.map((step) => ({ ...step, status: "complete" as const })),
                  }
                : job,
            ),
          );
          if (data.batch) setBatch(data.batch);
          return;
        }

        if (data.type === "cancelled" || data.type === "error") {
          setJobs((previous) =>
            previous.map((job) =>
              job.runId === data.runId
                ? {
                    ...job,
                    category: data.category ?? job.category,
                    recordNumber: data.recordNumber ?? job.recordNumber,
                    error: data.type === "cancelled" ? "Cancelled - same number ready to retry" : data.error,
                    cancelled: data.type === "cancelled",
                    steps: job.steps.map((step) =>
                      step.status === "processing" ? { ...step, status: "error" as const } : step,
                    ),
                  }
                : job,
            ),
          );
        }
      } catch (error) {
        console.error("LiveFeed: Failed to parse message", error);
      }
    };

    return () => eventSource.close();
  }, []);

  const configureBatch = async () => {
    setBatchError("");
    setSavingBatch(true);
    try {
      const response = await fetch("/api/validation-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category,
          start: Number(rangeStart),
          end: Number(rangeEnd),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not configure validation range.");
      setBatch(payload.batch);
      setJobs([]);
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : "Could not configure validation range.");
    } finally {
      setSavingBatch(false);
    }
  };

  const cancelJob = async (job: Job) => {
    const response = await fetch("/api/cancel-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: job.runId,
        category: job.category,
        recordNumber: job.recordNumber,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      setJobs((previous) =>
        previous.map((item) =>
          item.runId === job.runId ? { ...item, error: "Could not cancel this process." } : item,
        ),
      );
    }
  };

  const exportRows = jobs
    .flatMap((job) => job.result?.validation?.export_rows ?? [])
    .sort((a, b) => a.record_number - b.record_number);

  const downloadCsv = () => {
    const header = ["category", "record_number", "lead", "your_pcc", "your_rmse", "your_snr"];
    const lines = exportRows.map((row) =>
      [row.category, row.record_number, row.lead, metric(row.your_pcc), metric(row.your_rmse), metric(row.your_snr)]
        .map((value) => JSON.stringify(value))
        .join(","),
    );
    downloadText([header.join(","), ...lines].join("\n"), "ecg_validation_results.csv", "text/csv");
  };

  const downloadTxt = () => {
    const lines = exportRows.map((row) =>
      [row.category, row.record_number, row.lead, metric(row.your_pcc), metric(row.your_rmse), metric(row.your_snr)].join("\t"),
    );
    downloadText(
      ["category\trecord_number\tlead\tyour_pcc\tyour_rmse\tyour_snr", ...lines].join("\n"),
      "ecg_validation_results.txt",
      "text/plain",
    );
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm sm:p-7">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-teal-600">Phone validation range</p>
            <h2 className="mt-2 text-2xl font-bold text-gray-900">Choose what the next phone scan validates</h2>
            <p className="mt-1 text-sm text-gray-600">
              Phone uploads are assigned automatically. A number advances only after successful validation.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-4">
            <label className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              Category
              <select
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                disabled={Boolean(batch?.activeRunId)}
                className="mt-1 block h-11 w-full rounded-lg border border-gray-300 bg-white px-3 text-base font-medium text-gray-900"
              >
                <option>HB</option>
                <option>MI</option>
                <option>Normal</option>
                <option>PMI</option>
              </select>
            </label>
            <label className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              From
              <input
                type="number"
                min="1"
                value={rangeStart}
                onChange={(event) => setRangeStart(event.target.value)}
                disabled={Boolean(batch?.activeRunId)}
                className="mt-1 block h-11 w-full rounded-lg border border-gray-300 px-3 text-base text-gray-900"
              />
            </label>
            <label className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              To
              <input
                type="number"
                min="1"
                value={rangeEnd}
                onChange={(event) => setRangeEnd(event.target.value)}
                disabled={Boolean(batch?.activeRunId)}
                className="mt-1 block h-11 w-full rounded-lg border border-gray-300 px-3 text-base text-gray-900"
              />
            </label>
            <button
              onClick={configureBatch}
              disabled={savingBatch || Boolean(batch?.activeRunId)}
              className="mt-5 flex h-11 items-center justify-center gap-2 rounded-lg bg-teal-600 px-5 font-semibold text-white hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {savingBatch ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              Set range
            </button>
          </div>
        </div>

        {batchError ? <p className="mt-4 text-sm font-medium text-red-600">{batchError}</p> : null}

        <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-gray-100 pt-5">
          {batch ? (
            <>
              <span className="rounded-full bg-gray-900 px-4 py-2 text-sm font-bold text-white">
                {batch.complete ? `${batch.category} ${batch.start}-${batch.end} complete` : `Next: ${batch.category} ${batch.current}`}
              </span>
              <span className="text-sm text-gray-600">
                Range {batch.category} {batch.start}-{batch.end}
              </span>
              {batch.activeRunId ? (
                <span className="flex items-center gap-2 text-sm font-medium text-blue-600">
                  <Loader2 className="h-4 w-4 animate-spin" /> Processing from phone
                </span>
              ) : !batch.complete ? (
                <span className="text-sm font-medium text-teal-700">Ready for phone upload</span>
              ) : null}
            </>
          ) : (
            <span className="text-sm font-medium text-amber-700">Set a range before taking the next phone photo.</span>
          )}

          {exportRows.length > 0 ? (
            <div className="ml-auto flex gap-2">
              <button onClick={downloadCsv} className="flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">
                <Download className="h-4 w-4" /> CSV
              </button>
              <button onClick={downloadTxt} className="flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">
                <Download className="h-4 w-4" /> TXT
              </button>
            </div>
          ) : null}
        </div>
      </section>

      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-2xl font-bold text-gray-900">
          <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
          Live AI Pipeline Feed
        </h2>
        <span className="text-sm text-gray-500">{jobs.length} recent sessions</span>
      </div>

      {jobs.length === 0 ? (
        <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-white p-12 text-center">
          <Activity className="mx-auto mb-4 h-12 w-12 text-gray-300" />
          <h3 className="mb-2 text-xl font-semibold text-gray-900">Waiting For Phone Uploads</h3>
          <p className="text-gray-500">Set the range above, then capture the ECG from the phone page.</p>
        </div>
      ) : (
        jobs.map((job) => {
          const complete = job.steps[job.steps.length - 1].status === "complete";
          const active = !job.error && !complete;
          return (
            <article key={job.runId} className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-xl">
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-100 bg-gray-50 px-6 py-4">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="rounded-full bg-teal-600 px-3 py-1 text-base font-bold text-white">
                    {job.category && job.recordNumber ? `${job.category} ${job.recordNumber}` : "Unassigned"}
                  </span>
                  <span className="font-mono text-xs text-gray-400">ID: {job.runId.slice(0, 8)}</span>
                  <span className="text-sm text-gray-600">{new Date(job.timestamp).toLocaleTimeString()}</span>
                </div>

                <div className="flex items-center gap-3">
                  {job.error ? (
                    <span className={`flex items-center gap-2 text-sm font-medium ${job.cancelled ? "text-amber-700" : "text-red-600"}`}>
                      <AlertCircle className="h-4 w-4" /> {job.error}
                    </span>
                  ) : complete ? (
                    <span className="flex items-center gap-2 text-sm font-medium text-teal-600">
                      <CheckCircle2 className="h-4 w-4" /> Validation complete
                    </span>
                  ) : (
                    <span className="flex items-center gap-2 text-sm font-medium text-blue-600">
                      <Loader2 className="h-4 w-4 animate-spin" /> Processing...
                    </span>
                  )}
                  {active ? (
                    <button
                      onClick={() => cancelJob(job)}
                      className="flex items-center gap-2 rounded-lg bg-red-600 px-3 py-2 text-sm font-semibold text-white hover:bg-red-700"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" /> Cancel
                    </button>
                  ) : null}
                </div>
              </header>

              <div className="p-6">
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
                  {job.steps.map((step) => {
                    const Icon = step.icon;
                    return (
                      <div
                        key={step.id}
                        className={`relative flex flex-col items-center rounded-xl border p-4 transition-all ${
                          step.status === "complete"
                            ? "border-teal-100 bg-teal-50"
                            : step.status === "processing"
                              ? "border-blue-200 bg-blue-50 ring-2 ring-blue-100"
                              : step.status === "error"
                                ? "border-red-200 bg-red-50"
                                : "border-gray-100 bg-gray-50 grayscale"
                        }`}
                      >
                        <div className={`mb-3 rounded-lg p-2 ${
                          step.status === "complete"
                            ? "bg-teal-500 text-white"
                            : step.status === "error"
                              ? "bg-red-500 text-white"
                              : "bg-gray-200 text-gray-500"
                        }`}>
                          <Icon className={`h-5 w-5 ${step.status === "processing" ? "animate-pulse" : ""}`} />
                        </div>
                        <span className={`text-center text-xs font-bold uppercase tracking-wider ${
                          step.status === "complete" ? "text-teal-700" : step.status === "error" ? "text-red-700" : "text-gray-500"
                        }`}>
                          {step.label}
                        </span>
                        <button
                          type="button"
                          disabled={!step.image}
                          className={`group relative mt-4 aspect-square w-full overflow-hidden rounded-lg border border-gray-200 bg-white ${
                            step.image ? "cursor-zoom-in hover:border-teal-400 hover:shadow-md" : ""
                          }`}
                          onClick={() => {
                            if (step.image) setSelectedImage({ src: `/api/files/${step.image}`, label: step.label });
                          }}
                        >
                          {step.image ? (
                            <img src={`/api/files/${step.image}`} alt={step.label} className="h-full w-full object-cover transition-transform group-hover:scale-105" />
                          ) : step.status === "processing" ? (
                            <span className="flex h-full w-full items-center justify-center bg-blue-50">
                              <Loader2 className="h-8 w-8 animate-spin text-blue-400" />
                            </span>
                          ) : (
                            <span className="flex h-full w-full items-center justify-center bg-gray-100">
                              <ImageIcon className="h-8 w-8 text-gray-300" />
                            </span>
                          )}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            </article>
          );
        })
      )}

      {selectedImage ? (
        <div className="fixed inset-0 z-50 flex cursor-zoom-out flex-col items-center justify-center bg-black/90 backdrop-blur-md" onClick={() => setSelectedImage(null)}>
          <button className="absolute right-6 top-6 rounded-full p-3 text-white hover:bg-white/10 hover:text-red-400" onClick={() => setSelectedImage(null)}>
            <X className="h-8 w-8" />
          </button>
          <img
            src={selectedImage.src}
            alt={selectedImage.label}
            className="max-h-[82vh] max-w-[92vw] cursor-default rounded-xl border border-white/10 object-contain shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          />
          <div className="mt-5 rounded-full border border-white/10 bg-black/60 px-6 py-2 font-semibold text-white">
            {selectedImage.label}
          </div>
        </div>
      ) : null}
    </div>
  );
}
