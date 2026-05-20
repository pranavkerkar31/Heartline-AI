"use client";
import React, { useEffect, useState, useRef } from "react";
import { 
  Loader2, 
  CheckCircle2, 
  Image as ImageIcon, 
  Maximize2, 
  Scissors, 
  Layers, 
  Activity,
  AlertCircle,
  X
} from "lucide-react";

interface ProcessStep {
  id: string;
  label: string;
  status: "waiting" | "processing" | "complete" | "error";
  image?: string;
  icon: any;
}

interface Job {
  runId: string;
  timestamp: number;
  steps: ProcessStep[];
  error?: string;
}

export default function LiveFeed() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedImage, setSelectedImage] = useState<{ src: string; label: string } | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Close modal on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedImage(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    console.log("LiveFeed: Initializing SSE connection...");
    const eventSource = new EventSource("/api/events");

    eventSource.onopen = () => {
      console.log("LiveFeed: SSE Connection opened successfully");
    };

    eventSource.onerror = (err) => {
      console.error("LiveFeed: SSE Connection error:", err);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("LiveFeed: Received data:", data);

      if (data.type === "start") {
        const newJob: Job = {
          runId: data.runId,
          timestamp: data.timestamp,
          steps: [
            { id: "received", label: "Received Photo", status: "complete", icon: ImageIcon },
            { id: "orientation", label: "Orientation Fix", status: "waiting", icon: CheckCircle2 },
            { id: "yolo", label: "YOLO Detection", status: "waiting", icon: Maximize2 },
            { id: "crop", label: "ECG Cropping", status: "waiting", icon: Scissors },
            { id: "enhanced", label: "AI Enhancement", status: "waiting", icon: Layers },
            { id: "mask", label: "Segmentation", status: "waiting", icon: Activity },
            { id: "digitized", label: "Digitizer Output", status: "waiting", icon: CheckCircle2 },
          ],
        };
        setJobs((prev) => [newJob, ...prev].slice(0, 5)); // Keep last 5 jobs
      } else if (data.type === "progress") {
        setJobs((prev) => {
          return prev.map((job) => {
            if (job.runId === data.runId) {
              const newSteps = job.steps.map((step) => {
                if (step.id === data.step) {
                  return { ...step, status: "complete" as const, image: data.image };
                }
                // Mark current and previous steps
                const stepIndex = job.steps.findIndex(s => s.id === step.id);
                const currentStepIndex = job.steps.findIndex(s => s.id === data.step);
                
                if (stepIndex < currentStepIndex) {
                  return { ...step, status: "complete" as const };
                }
                if (stepIndex === currentStepIndex + 1) {
                  return { ...step, status: "processing" as const };
                }
                return step;
              });
              return { ...job, steps: newSteps };
            }
            return job;
          });
        });
      } else if (data.type === "error") {
        setJobs((prev) => {
          return prev.map((job) => {
            if (job.runId === data.runId) {
              return { ...job, error: data.error };
            }
            return job;
          });
        });
        }
      } catch (e) {
        console.error("LiveFeed: Failed to parse message", e);
      }
    };

    return () => eventSource.close();
  }, []);

  if (jobs.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-12 border-2 border-dashed border-gray-200 text-center">
        <Activity className="w-12 h-12 text-gray-300 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-900 mb-2">Live Processing Feed</h3>
        <p className="text-gray-500">Capture an ECG from the mobile app to see the AI pipeline in action.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          Live AI Pipeline Feed
        </h2>
        <span className="text-sm text-gray-500">{jobs.length} active sessions</span>
      </div>

      {jobs.map((job) => (
        <div key={job.runId} className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transition-all hover:shadow-2xl">
          {/* Job Header */}
          <div className="bg-gray-50 px-6 py-4 border-b border-gray-100 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-gray-400">ID: {job.runId.slice(0, 8)}</span>
              <span className="text-sm text-gray-600">
                {new Date(job.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {job.error ? (
              <div className="flex items-center gap-2 text-red-600 text-sm font-medium">
                <AlertCircle className="w-4 h-4" />
                Error occurred
              </div>
            ) : job.steps[job.steps.length - 1].status === "complete" ? (
              <div className="flex items-center gap-2 text-teal-600 text-sm font-medium">
                <CheckCircle2 className="w-4 h-4" />
                Pipeline Complete
              </div>
            ) : (
              <div className="flex items-center gap-2 text-blue-600 text-sm font-medium">
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </div>
            )}
          </div>

          {/* Steps Grid */}
          <div className="p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6">
              {job.steps.map((step) => {
                const Icon = step.icon;
                return (
                  <div 
                    key={step.id} 
                    className={`relative flex flex-col items-center p-4 rounded-xl border transition-all ${
                      step.status === "complete" 
                        ? "bg-teal-50 border-teal-100" 
                        : step.status === "processing"
                        ? "bg-blue-50 border-blue-200 ring-2 ring-blue-100"
                        : "bg-gray-50 border-gray-100 grayscale"
                    }`}
                  >
                    <div className={`p-2 rounded-lg mb-3 ${
                      step.status === "complete" ? "bg-teal-500 text-white" : "bg-gray-200 text-gray-500"
                    }`}>
                      <Icon className={`w-5 h-5 ${step.status === "processing" ? "animate-pulse" : ""}`} />
                    </div>
                    
                    <span className={`text-xs font-bold text-center uppercase tracking-wider ${
                      step.status === "complete" ? "text-teal-700" : "text-gray-500"
                    }`}>
                      {step.label}
                    </span>

                    {/* Step Image */}
                    <div 
                      className={`mt-4 w-full aspect-square bg-white rounded-lg border border-gray-200 overflow-hidden relative group transition-all duration-300 ${
                        step.image 
                          ? "cursor-zoom-in hover:border-teal-400 hover:shadow-md" 
                          : ""
                      }`}
                      onClick={() => {
                        if (step.image) {
                          setSelectedImage({
                            src: `/api/files/${step.image}`,
                            label: step.label
                          });
                        }
                      }}
                    >
                      {step.image ? (
                        <>
                          <img 
                            src={`/api/files/${step.image}`} 
                            alt={step.label}
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                          />
                          {/* Hover Overlay */}
                          <div className="absolute inset-0 bg-black/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <div className="bg-white/90 backdrop-blur-sm p-1.5 rounded-lg shadow-sm">
                              <Maximize2 className="w-4 h-4 text-gray-700" />
                            </div>
                          </div>
                        </>
                      ) : step.status === "processing" ? (
                        <div className="w-full h-full flex items-center justify-center bg-blue-50">
                          <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
                        </div>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-gray-100">
                          <ImageIcon className="w-8 h-8 text-gray-300" />
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {job.error && (
            <div className="px-6 py-3 bg-red-50 border-t border-red-100">
              <p className="text-sm text-red-600 font-mono">{job.error}</p>
            </div>
          )}
        </div>
      ))}

      {/* Lightbox / Maximized Image Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/90 backdrop-blur-md animate-in fade-in duration-200 cursor-zoom-out"
          onClick={() => setSelectedImage(null)}
        >
          {/* Close Button */}
          <button 
            className="absolute top-6 right-6 text-white hover:text-red-400 p-3 rounded-full hover:bg-white/10 transition-all duration-200 z-50 cursor-pointer"
            onClick={(e) => {
              e.stopPropagation();
              setSelectedImage(null);
            }}
          >
            <X className="w-8 h-8" />
          </button>

          {/* Image Wrapper to prevent click propagation outside image */}
          <div 
            className="relative max-w-[90vw] max-h-[80vh] flex items-center justify-center p-2 animate-in zoom-in-95 duration-200 cursor-default"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={selectedImage.src} 
              alt={selectedImage.label}
              className="max-w-full max-h-[80vh] object-contain rounded-xl shadow-2xl border border-white/10"
            />
          </div>

          {/* Caption */}
          <div className="mt-6 bg-black/60 border border-white/10 backdrop-blur-sm text-white px-6 py-2 rounded-full text-base font-semibold tracking-wide shadow-xl select-none">
            {selectedImage.label}
          </div>
        </div>
      )}
    </div>
  );
}
