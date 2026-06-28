"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle,
  Download,
  Info,
  Play,
  RotateCcw,
  Square,
  Upload,
  X,
} from "lucide-react";

type LeadMetric = {
  lead: string;
  your_pcc: number | null;
  your_rmse: number | null;
  your_snr: number | null;
  offset_ms: number | null;
  time_scale: number | null;
  match_score: number | null;
  samples: number;
};

type ExportRow = {
  category: string;
  record_number: number;
  lead: string;
  your_pcc: number | null;
  your_rmse: number | null;
  your_snr: number | null;
};

type ValidationSummary = {
  lead_count: number;
  mean_pcc: number | null;
  mean_rmse: number | null;
  mean_snr: number | null;
};

type ValidationResult = {
  category: string;
  record_number: number;
  truth_npz_path: string;
  comparison_image: string;
  lead_metrics: LeadMetric[];
  export_rows: ExportRow[];
  summary: ValidationSummary;
};

type AnalysisResult = {
  success: boolean;
  run_id: string;
  width: number;
  height: number;
  processed_image: string;
  cropped_image: string;
  npz_file: string;
  validation?: ValidationResult;
  message?: string;
};

type BatchRecord = {
  runId: string;
  category: string;
  recordNumber: number;
  validation: ValidationResult;
};

type ProgressEvent = {
  type: string;
  step?: string;
  runId?: string;
  image?: string;
  category?: string | null;
  recordNumber?: string | number | null;
  error?: string;
};

type AttemptRef = {
  category: string;
  recordNumber: number;
} | null;

const CATEGORY_OPTIONS = ["HB", "MI", "Normal", "PMI"];

const STEP_LABELS: Record<string, string> = {
  received: "Upload received",
  orientation: "Orientation checked",
  yolo: "Detection preview ready",
  crop: "ECG cropped",
  enhanced: "Image enhanced",
  mask: "Waveform mask built",
  digitized: "Signals reconstructed",
  validation: "Validation complete",
};

function formatMetric(value: number | null, digits: number) {
  return value == null || Number.isNaN(value) ? "--" : value.toFixed(digits);
}

function buildFileUrl(relativePath: string | null | undefined) {
  if (!relativePath) return null;
  const parts = relativePath.split("/").filter(Boolean).map(encodeURIComponent);
  return `/api/files/${parts.join("/")}`;
}

function triggerDownload(contents: string, filename: string, mimeType: string) {
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export default function CameraPage() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [category, setCategory] = useState("MI");
  const [rangeStart, setRangeStart] = useState("1");
  const [rangeEnd, setRangeEnd] = useState("20");
  const [batchStarted, setBatchStarted] = useState(false);
  const [currentRecord, setCurrentRecord] = useState<number | null>(null);
  const [batchEndRecord, setBatchEndRecord] = useState<number | null>(null);
  const [completedRecords, setCompletedRecords] = useState<BatchRecord[]>([]);
  const [selectedResultRunId, setSelectedResultRunId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingRunId, setProcessingRunId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Set a category and range to begin validation.");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [progressImages, setProgressImages] = useState<Record<string, string>>({});
  const [progressSteps, setProgressSteps] = useState<string[]>([]);

  const captureInputRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const activeAttemptRef = useRef<AttemptRef>(null);
  const processingRunIdRef = useRef<string | null>(null);

  const selectedRecord = useMemo(() => {
    return completedRecords.find((record) => record.runId === selectedResultRunId) ?? completedRecords.at(-1) ?? null;
  }, [completedRecords, selectedResultRunId]);

  const totalPlanned = useMemo(() => {
    if (!batchStarted || currentRecord == null || batchEndRecord == null) return 0;
    return batchEndRecord - Number(rangeStart) + 1;
  }, [batchStarted, batchEndRecord, currentRecord, rangeStart]);

  const batchRows = useMemo(() => {
    return completedRecords.flatMap((record) => record.validation.export_rows);
  }, [completedRecords]);

  useEffect(() => {
    const source = new EventSource("/api/events");

    source.onmessage = (event) => {
      const data = JSON.parse(event.data) as ProgressEvent;
      const expectedAttempt = activeAttemptRef.current;

      if (data.type === "start" && expectedAttempt) {
        const incomingRecord = Number(data.recordNumber);
        if (data.category === expectedAttempt.category && incomingRecord === expectedAttempt.recordNumber) {
          processingRunIdRef.current = data.runId ?? null;
          setProcessingRunId(data.runId ?? null);
          setStatusMessage(`Processing ${expectedAttempt.category} ${expectedAttempt.recordNumber}...`);
        }
        return;
      }

      if (!data.runId || data.runId !== processingRunIdRef.current) return;

      if (data.type === "progress" && data.step) {
        const label = STEP_LABELS[data.step] ?? data.step;
        setProgressSteps((prev) => (prev.includes(label) ? prev : [...prev, label]));
        setStatusMessage(label);
        if (data.image) {
          setProgressImages((prev) => ({ ...prev, [data.step!]: data.image! }));
        }
      }

      if (data.type === "cancelled") {
        setStatusMessage(`Cancelled ${category} ${currentRecord ?? ""}. Ready to retry the same record.`);
        setIsProcessing(false);
        processingRunIdRef.current = null;
        setProcessingRunId(null);
        activeAttemptRef.current = null;
      }

      if (data.type === "error") {
        setErrorMessage(data.error ?? "Processing failed.");
      }
    };

    source.onerror = () => {
      setStatusMessage((prev) => prev || "Live progress is temporarily unavailable.");
    };

    return () => source.close();
  }, [category, currentRecord, processingRunId]);

  const handleImageSelection = (file: File | null) => {
    if (!file) return;
    setUploadedFile(file);
    setErrorMessage(null);
    const reader = new FileReader();
    reader.onload = (event) => setUploadedImage(event.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleImageSelection(event.target.files?.[0] ?? null);
    event.target.value = "";
  };

  const handleClearImage = () => {
    setUploadedImage(null);
    setUploadedFile(null);
  };

  const startBatch = () => {
    const start = Number(rangeStart);
    const end = Number(rangeEnd);

    if (!Number.isInteger(start) || !Number.isInteger(end) || start <= 0 || end < start) {
      setErrorMessage("Enter a valid inclusive range such as 34 to 55.");
      return;
    }

    setBatchStarted(true);
    setCurrentRecord(start);
    setBatchEndRecord(end);
    setCompletedRecords([]);
    setSelectedResultRunId(null);
    setUploadedImage(null);
    setUploadedFile(null);
    setProgressImages({});
    setProgressSteps([]);
    processingRunIdRef.current = null;
    setProcessingRunId(null);
    setIsProcessing(false);
    setErrorMessage(null);
    setStatusMessage(`Batch ready. Current target: ${category} ${start}.`);
  };

  const resetBatch = () => {
    setBatchStarted(false);
    setCurrentRecord(null);
    setBatchEndRecord(null);
    setCompletedRecords([]);
    setSelectedResultRunId(null);
    setUploadedImage(null);
    setUploadedFile(null);
    setProgressImages({});
    setProgressSteps([]);
    processingRunIdRef.current = null;
    setProcessingRunId(null);
    setIsProcessing(false);
    setErrorMessage(null);
    setStatusMessage("Set a category and range to begin validation.");
    activeAttemptRef.current = null;
  };

  const retryLastCompleted = () => {
    const last = completedRecords.at(-1);
    if (!last || isProcessing) return;

    setCompletedRecords((prev) => prev.slice(0, -1));
    setCurrentRecord(last.recordNumber);
    setSelectedResultRunId(completedRecords.at(-2)?.runId ?? null);
    setUploadedImage(null);
    setUploadedFile(null);
    setProgressImages({});
    setProgressSteps([]);
    setStatusMessage(`Retry ${last.category} ${last.recordNumber} with a better capture.`);
  };

  const handleAnalysis = async () => {
    if (!uploadedFile) {
      setErrorMessage("Capture or upload an ECG image first.");
      return;
    }
    if (!batchStarted || currentRecord == null || batchEndRecord == null) {
      setErrorMessage("Start a batch before running validation.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setProcessingRunId(null);
    setProgressImages({});
    setProgressSteps([]);
    setStatusMessage(`Uploading ${category} ${currentRecord}...`);
    activeAttemptRef.current = { category, recordNumber: currentRecord };

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);
      formData.append("category", category);
      formData.append("recordNumber", String(currentRecord));

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (!response.ok || !result.success) {
        if (result.cancelled) {
          setStatusMessage(`Cancelled ${category} ${currentRecord}. Capture the same record again when ready.`);
          handleClearImage();
          return;
        }
        throw new Error(result.error || "Processing failed");
      }

      const analysis = result as AnalysisResult;
      if (!analysis.validation) {
        throw new Error("Validation output was missing from the response.");
      }

      const completedRecord: BatchRecord = {
        runId: analysis.run_id,
        category,
        recordNumber: currentRecord,
        validation: analysis.validation,
      };

      setCompletedRecords((prev) => [...prev, completedRecord]);
      setSelectedResultRunId(analysis.run_id);
      handleClearImage();

      if (currentRecord >= batchEndRecord) {
        setStatusMessage(`Batch complete. Processed ${category} ${rangeStart}-${batchEndRecord}.`);
        setCurrentRecord(batchEndRecord + 1);
      } else {
        const nextRecord = currentRecord + 1;
        setCurrentRecord(nextRecord);
        setStatusMessage(`Validated ${category} ${currentRecord}. Next target: ${category} ${nextRecord}.`);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to analyze image.");
      setStatusMessage(`Validation for ${category} ${currentRecord} did not complete.`);
    } finally {
      setIsProcessing(false);
      processingRunIdRef.current = null;
      setProcessingRunId(null);
      activeAttemptRef.current = null;
    }
  };

  const cancelCurrentRun = async () => {
    if (!processingRunId || currentRecord == null) return;

    try {
      await fetch("/api/cancel-run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runId: processingRunId, category, recordNumber: currentRecord }),
      });
      handleClearImage();
      setStatusMessage(`Cancelling ${category} ${currentRecord}...`);
    } catch {
      setErrorMessage("Could not cancel the current run.");
    }
  };

  const downloadCsv = () => {
    const header = ["category", "record_number", "lead", "your_pcc", "your_rmse", "your_snr"];
    const lines = [header.join(",")];
    for (const row of batchRows) {
      lines.push(
        [
          row.category,
          row.record_number,
          row.lead,
          row.your_pcc ?? "",
          row.your_rmse ?? "",
          row.your_snr ?? "",
        ].join(",")
      );
    }
    triggerDownload(lines.join("\n"), `${category}_${rangeStart}-${rangeEnd}_validation.csv`, "text/csv;charset=utf-8");
  };

  const downloadTxt = () => {
    const lines = ["category	record_number	lead	your_pcc	your_rmse	your_snr"];
    for (const row of batchRows) {
      lines.push(
        [
          row.category,
          row.record_number,
          row.lead,
          row.your_pcc ?? "",
          row.your_rmse ?? "",
          row.your_snr ?? "",
        ].join("	")
      );
    }
    triggerDownload(lines.join("\n"), `${category}_${rangeStart}-${rangeEnd}_validation.txt`, "text/plain;charset=utf-8");
  };

  const comparisonImageUrl = buildFileUrl(selectedRecord?.validation.comparison_image);
  const currentTargetComplete = batchStarted && currentRecord != null && batchEndRecord != null && currentRecord > batchEndRecord;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1.15fr_0.85fr] gap-6 lg:gap-8">
      <div className="space-y-6">
        <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Batch Validation Setup</h2>
          <p className="text-gray-600 mb-6">Choose the dataset category and record range you want to validate against while scanning from your phone.</p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
            <label className="block">
              <span className="text-sm font-medium text-gray-700">Category</span>
              <select
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                disabled={batchStarted && completedRecords.length > 0}
                className="mt-2 w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:border-teal-500 focus:outline-none"
              >
                {CATEGORY_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-sm font-medium text-gray-700">Start</span>
              <input
                value={rangeStart}
                onChange={(event) => setRangeStart(event.target.value)}
                inputMode="numeric"
                className="mt-2 w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:border-teal-500 focus:outline-none"
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-gray-700">End</span>
              <input
                value={rangeEnd}
                onChange={(event) => setRangeEnd(event.target.value)}
                inputMode="numeric"
                className="mt-2 w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:border-teal-500 focus:outline-none"
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={startBatch}
              className="bg-teal-500 hover:bg-teal-600 text-white px-5 py-2.5 rounded-lg font-medium transition-colors"
            >
              {batchStarted ? "Restart Batch" : "Start Batch"}
            </button>
            <button
              onClick={resetBatch}
              className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-5 py-2.5 rounded-lg font-medium transition-colors"
            >
              Reset
            </button>
            <button
              onClick={retryLastCompleted}
              disabled={completedRecords.length === 0 || isProcessing}
              className="bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 text-gray-700 border border-gray-300 px-5 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Retry Last Record
            </button>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-teal-200 bg-teal-50 p-4">
              <p className="text-sm text-teal-700">Current Target</p>
              <p className="text-2xl font-semibold text-teal-950">
                {currentTargetComplete ? "Done" : currentRecord != null ? `${category} ${currentRecord}` : "--"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-semibold text-gray-900">{completedRecords.length}</p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
              <p className="text-sm text-gray-600">Planned</p>
              <p className="text-2xl font-semibold text-gray-900">{totalPlanned || "--"}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Capture Current ECG</h2>
          <p className="text-gray-600 mb-6">The batch only advances after a successful validation. If you cancel a bad scan, the same record stays active.</p>

          <div className="border-2 border-dashed border-teal-300 rounded-xl p-4 sm:p-6 mb-6 bg-gray-50 relative">
            {uploadedImage ? (
              <div className="relative">
                <img src={uploadedImage} alt="Uploaded ECG" className="w-full h-auto rounded-lg" />
                <button
                  onClick={handleClearImage}
                  className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="border-2 border-dashed border-teal-400 rounded-lg p-8 sm:p-16 flex flex-col items-center justify-center">
                <Camera className="w-16 h-16 text-gray-400 mb-4" />
                <p className="text-gray-700 font-medium mb-1">Ready for {currentRecord != null ? `${category} ${currentRecord}` : "the next record"}</p>
                <p className="text-gray-500 text-sm">Capture from your phone camera or upload an existing image.</p>
              </div>
            )}
          </div>

          <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 mb-6 flex items-start gap-3">
            <Info className="w-5 h-5 text-teal-600 mt-0.5 shrink-0" />
            <p className="text-teal-800 text-sm">Keep the ECG flat, include the full sheet, and avoid glare. If this attempt looks bad, cancel it and recapture the same record.</p>
          </div>

          {errorMessage && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 shrink-0" />
              <p className="text-red-800 text-sm">{errorMessage}</p>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <button
              onClick={() => captureInputRef.current?.click()}
              className="bg-teal-500 hover:bg-teal-600 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
            >
              <Camera className="w-5 h-5" />
              Capture Photo
            </button>
            <button
              onClick={() => uploadInputRef.current?.click()}
              className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
            >
              <Upload className="w-5 h-5" />
              Upload Image
            </button>
          </div>

          <input ref={captureInputRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleFileInput} />
          <input ref={uploadInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileInput} />

          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleAnalysis}
              disabled={isProcessing || !uploadedFile || currentTargetComplete || !batchStarted}
              className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium flex items-center gap-2 transition-colors"
            >
              <Play className="w-5 h-5" />
              {isProcessing ? "Processing..." : `Validate ${currentRecord != null ? `${category} ${currentRecord}` : "Record"}`}
            </button>
            <button
              onClick={cancelCurrentRun}
              disabled={!isProcessing || !processingRunId}
              className="bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 text-gray-700 border border-gray-300 px-6 py-3 rounded-lg font-medium flex items-center gap-2 transition-colors"
            >
              <Square className="w-4 h-4" />
              Cancel Current Run
            </button>
          </div>

          <div className="mt-5 rounded-lg border border-gray-200 bg-gray-50 px-4 py-3">
            <p className="text-sm text-gray-700">{statusMessage}</p>
          </div>

          {progressSteps.length > 0 && (
            <div className="mt-5">
              <p className="text-sm font-medium text-gray-800 mb-3">Processing steps</p>
              <div className="flex flex-wrap gap-2">
                {progressSteps.map((step) => (
                  <span key={step} className="inline-flex items-center rounded-full bg-teal-100 px-3 py-1 text-xs font-medium text-teal-800">
                    {step}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {selectedRecord && (
          <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-gray-900">Validation Output</h2>
                <p className="text-gray-600">Lead-wise comparison for {selectedRecord.category} {selectedRecord.recordNumber}</p>
              </div>
              <div className="text-sm text-gray-600">
                Mean PCC {formatMetric(selectedRecord.validation.summary.mean_pcc, 3)} | Mean RMSE {formatMetric(selectedRecord.validation.summary.mean_rmse, 3)} | Mean SNR {formatMetric(selectedRecord.validation.summary.mean_snr, 2)}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 text-left text-gray-600">
                    <th className="py-3 pr-4 font-medium">Lead</th>
                    <th className="py-3 pr-4 font-medium">Your PCC</th>
                    <th className="py-3 pr-4 font-medium">Your RMSE</th>
                    <th className="py-3 pr-4 font-medium">Your SNR</th>
                    <th className="py-3 pr-4 font-medium">Offset ms</th>
                    <th className="py-3 pr-4 font-medium">Scale</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRecord.validation.lead_metrics.map((row) => (
                    <tr key={`${selectedRecord.runId}-${row.lead}`} className="border-b border-gray-100 text-gray-800">
                      <td className="py-3 pr-4 font-medium">{row.lead}</td>
                      <td className="py-3 pr-4">{formatMetric(row.your_pcc, 3)}</td>
                      <td className="py-3 pr-4">{formatMetric(row.your_rmse, 3)}</td>
                      <td className="py-3 pr-4">{formatMetric(row.your_snr, 2)}</td>
                      <td className="py-3 pr-4">{formatMetric(row.offset_ms, 0)}</td>
                      <td className="py-3 pr-4">{formatMetric(row.time_scale, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {comparisonImageUrl && (
              <div className="mt-6">
                <p className="text-sm font-medium text-gray-800 mb-3">Debug waveform comparison</p>
                <img src={comparisonImageUrl} alt="Waveform comparison" className="w-full rounded-xl border border-gray-200" />
              </div>
            )}
          </div>
        )}
      </div>

      <div className="space-y-6">
        <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-2xl font-semibold text-gray-900">Batch Results</h2>
              <p className="text-gray-600">Download all completed validations after the range is done, or anytime during review.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={downloadCsv}
                disabled={batchRows.length === 0}
                className="bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 text-gray-700 border border-gray-300 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                CSV
              </button>
              <button
                onClick={downloadTxt}
                disabled={batchRows.length === 0}
                className="bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 text-gray-700 border border-gray-300 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                TXT
              </button>
            </div>
          </div>

          {completedRecords.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-300 p-6 text-sm text-gray-500">
              Completed records will appear here as each scan is validated. Export files include the requested columns only: lead, your PCC, your RMSE, and your SNR, plus category and record number for batch tracking.
            </div>
          ) : (
            <div className="space-y-3">
              {completedRecords.map((record) => (
                <button
                  key={record.runId}
                  onClick={() => setSelectedResultRunId(record.runId)}
                  className={`w-full text-left rounded-xl border px-4 py-3 transition-colors ${selectedResultRunId === record.runId ? "border-teal-500 bg-teal-50" : "border-gray-200 hover:bg-gray-50"}`}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="font-medium text-gray-900">{record.category} {record.recordNumber}</p>
                      <p className="text-sm text-gray-600">{record.validation.summary.lead_count} leads | mean PCC {formatMetric(record.validation.summary.mean_pcc, 3)}</p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-teal-600" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Capture Guidelines</h2>
          <p className="text-gray-600 mb-6">A few checks help the validator align the paper trace with the digital ground truth much more reliably.</p>

          <div className="space-y-4 text-sm text-gray-700">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
              <p>Place the ECG sheet on a flat surface and keep all 12 leads fully visible.</p>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
              <p>Capture in landscape with the camera roughly parallel to the paper.</p>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
              <p>Avoid strong shadows, reflections, and clipped edges because they hurt signal extraction.</p>
            </div>
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
              <p>If a scan looks poor, cancel it before completion or use Retry Last Record right after completion to keep the same dataset index active.</p>
            </div>
          </div>

          {Object.keys(progressImages).length > 0 && (
            <div className="mt-6">
              <p className="text-sm font-medium text-gray-800 mb-3">Latest debug previews</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {Object.entries(progressImages).map(([step, relativePath]) => {
                  const imageUrl = buildFileUrl(relativePath);
                  if (!imageUrl) return null;
                  return (
                    <div key={step} className="rounded-lg border border-gray-200 overflow-hidden">
                      <div className="px-3 py-2 bg-gray-50 text-xs font-medium text-gray-600 uppercase tracking-wide">{STEP_LABELS[step] ?? step}</div>
                      <img src={imageUrl} alt={step} className="w-full h-auto" />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
