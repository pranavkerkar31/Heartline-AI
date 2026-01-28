"use client";
import React, { useState, useRef } from "react";
import { Camera, Upload, CheckCircle, Info, X, Play } from "lucide-react";

export default function CameraPage() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stableFrames = useRef(0);

  const [scanning, setScanning] = useState(false);
  const [scanText, setScanText] = useState("Looking for ECG paper...");

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => setUploadedImage(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleClearImage = () => {
    setUploadedImage(null);
    setUploadedFile(null);
    setAnalysisResult(null);
  };

  const startCameraScan = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setScanning(true);
        requestAnimationFrame(scanLoop);
      }
    } catch (error) {
      console.error("Camera access error:", error);
      alert("Unable to access camera. Please check permissions.");
    }
  };

  const scanLoop = async () => {
    if (!videoRef.current || !canvasRef.current || !scanning) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    const blob = await new Promise<Blob>((res) =>
      canvas.toBlob((b) => res(b!), "image/jpeg")
    );

    try {
      const form = new FormData();
      form.append("file", blob);

      const r = await fetch("http://127.0.0.1:8000/detect-paper", {
        method: "POST",
        body: form,
      });

      const data = await r.json();

      if (!data.found) {
        stableFrames.current = 0;
        setScanText("Looking for ECG paper...");
      } else {
        setScanText("Hold steady...");
        stableFrames.current++;

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 4;
        ctx.beginPath();
        data.corners.forEach(([x, y]: number[], i: number) => {
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();

        if (stableFrames.current >= 10) {
          captureFinal(blob);
          return;
        }
      }
    } catch (error) {
      console.error("Detection error:", error);
    }

    requestAnimationFrame(scanLoop);
  };

  const captureFinal = async (blob: Blob) => {
    setScanning(false);

    try {
      const form = new FormData();
      form.append("file", blob);

      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: form,
      });

      const result = await res.json();
      setAnalysisResult(result);

      // Stop camera stream
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }

      alert("ECG captured and processed successfully!");
    } catch (error) {
      console.error("Capture error:", error);
      alert("Failed to process image. Please try again.");
    }
  };

  const handleAnalysis = async () => {
    if (!uploadedFile) return;

    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error("Analysis error:", error);
      alert("Failed to analyze image. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
      {/* Capture/Upload Section */}
      <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          Capture or Upload ECG
        </h2>
        <p className="text-gray-600 mb-6">
          Start by capturing a photo or uploading an existing ECG image
        </p>

        {/* Camera Preview / Uploaded Image */}
        <div className="border-2 border-dashed border-teal-300 rounded-xl p-4 sm:p-6 mb-6 bg-gray-50 relative">
          {scanning ? (
            <>
              <video ref={videoRef} className="w-full h-auto rounded-lg" />
              <canvas
                ref={canvasRef}
                className="absolute top-4 left-4 right-4 bottom-4 w-[calc(100%-2rem)] h-[calc(100%-2rem)] pointer-events-none"
              />
              <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-lg text-sm font-medium">
                {scanText}
              </div>
            </>
          ) : uploadedImage ? (
            <div className="relative">
              <img
                src={uploadedImage}
                alt="Uploaded ECG"
                className="w-full h-auto rounded-lg"
              />
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
              <p className="text-gray-700 font-medium mb-1">Camera Preview</p>
              <p className="text-gray-500 text-sm">
                Position your ECG paper here
              </p>
            </div>
          )}
        </div>

        {/* Instruction */}
        {!uploadedImage && !scanning && (
          <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 mb-6 flex items-start space-x-3">
            <Info className="w-5 h-5 text-teal-600 mt-0.5 shrink-0" />
            <p className="text-teal-800 text-sm">
              Align the ECG paper horizontally within the frame
            </p>
          </div>
        )}

        {/* Action Buttons */}
        {!uploadedImage && !scanning ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <button
              onClick={startCameraScan}
              className="bg-teal-500 hover:bg-teal-600 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 cursor-pointer transition-colors"
            >
              <Camera className="w-5 h-5" />
              <span>Start Camera Scan</span>
            </button>
            <label className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 cursor-pointer transition-colors">
              <Upload className="w-5 h-5" />
              <span>Upload Image</span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
            </label>
          </div>
        ) : uploadedImage ? (
          <div className="space-y-3">
            <button
              onClick={handleAnalysis}
              disabled={isProcessing}
              className="w-full bg-teal-500 hover:bg-teal-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 shadow-md hover:shadow-lg transition-all"
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Run Analysis</span>
                </>
              )}
            </button>
            {analysisResult && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <p className="text-green-800 font-medium mb-2">
                  Analysis Complete!
                </p>
                <p className="text-sm text-green-700">
                  Width: {analysisResult.width}px, Height:{" "}
                  {analysisResult.height}px
                </p>
              </div>
            )}
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={startCameraScan}
                className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-4 py-2 rounded-lg font-medium flex items-center justify-center space-x-2 text-sm transition-colors"
              >
                <Camera className="w-4 h-4" />
                <span>Recapture</span>
              </button>
              <label className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-4 py-2 rounded-lg font-medium flex items-center justify-center space-x-2 cursor-pointer text-sm transition-colors">
                <Upload className="w-4 h-4" />
                <span>Re-upload</span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageUpload}
                />
              </label>
            </div>
          </div>
        ) : null}
      </div>

      {/* Guidelines Section */}
      <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          Capture Guidelines
        </h2>
        <p className="text-gray-600 mb-6">
          Follow these best practices for optimal results
        </p>

        <div className="space-y-5">
          {/* Guideline 1 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">
                Place ECG paper on a flat surface
              </p>
              <p className="text-sm text-gray-600">
                Avoid wrinkles or folds in the paper
              </p>
            </div>
          </div>

          {/* Guideline 2 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">
                Capture image in landscape orientation
              </p>
              <p className="text-sm text-gray-600">
                Horizontal format works best
              </p>
            </div>
          </div>

          {/* Guideline 3 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">
                Ensure the full ECG sheet is visible
              </p>
              <p className="text-sm text-gray-600">
                Include all 12 leads in the frame
              </p>
            </div>
          </div>

          {/* Guideline 4 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">
                Keep the camera parallel to the paper
              </p>
              <p className="text-sm text-gray-600">
                Avoid angled or tilted shots
              </p>
            </div>
          </div>

          {/* Guideline 5 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">
                Avoid strong shadows and glare
              </p>
              <p className="text-sm text-gray-600">
                Use diffused lighting when possible
              </p>
            </div>
          </div>

          {/* Guideline 6 */}
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-gray-900">Ensure good lighting</p>
              <p className="text-sm text-gray-600">
                Natural light or bright indoor lighting
              </p>
            </div>
          </div>
        </div>

        {/* Pro Tip */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-600 mt-0.5 shrink-0" />
          <div>
            <p className="font-medium text-blue-900 mb-1">Pro Tip</p>
            <p className="text-sm text-blue-800">
              For best results, clean the ECG paper surface and ensure there are
              no obstructions or annotations that might interfere with the
              analysis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}