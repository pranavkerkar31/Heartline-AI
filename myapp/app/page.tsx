"use client"
import React, { useState } from 'react';
import { Camera, Upload, Zap, Shield, TrendingUp, CheckCircle, Info, HelpCircle, X, Play } from 'lucide-react';

export default function ECGDigitisation() {
  const [activeTab, setActiveTab] = useState('home');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const handleCameraCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-teal-500 rounded-lg p-2">
                <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-xl font-semibold text-gray-900">Heartline AI</span>
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-gray-700 hover:text-gray-900 font-medium">Login</button>
              <button className="bg-teal-500 hover:bg-teal-600 text-white px-4 py-2 rounded-lg font-medium flex items-center space-x-2">
                <span>Sign Up</span>
              </button>
         
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {activeTab === 'home' && (
          <>
            {/* Hero Section */}
            <div className="text-center mb-12 sm:mb-16">
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
                ECG Digitisation & AI Analysis
              </h1>
              <p className="text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto">
                Transform traditional ECG paper records into digital format with AI-powered analysis. Detect heart conditions quickly and accurately using advanced machine learning.
              </p>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              {/* Fast Processing */}
              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="bg-teal-100 rounded-lg p-3 w-fit mb-4">
                  <Zap className="w-6 h-6 text-teal-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Fast Processing</h3>
                <p className="text-gray-600">AI-powered analysis in seconds</p>
              </div>

              {/* Accurate Detection */}
              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="bg-blue-100 rounded-lg p-3 w-fit mb-4">
                  <Shield className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Accurate Detection</h3>
                <p className="text-gray-600">CNN-based classification system</p>
              </div>

              {/* Advanced Analytics */}
              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="bg-purple-100 rounded-lg p-3 w-fit mb-4">
                  <TrendingUp className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Advanced Analytics</h3>
                <p className="text-gray-600">Comprehensive heart health insights</p>
              </div>
            </div>

            {/* CTA Button */}
            <div className="text-center">
              <button 
                onClick={() => setActiveTab('capture')}
                className="bg-teal-500 hover:bg-teal-600 text-white px-8 py-3 rounded-lg font-medium text-lg shadow-md hover:shadow-lg transition-all"
              >
                Get Started
              </button>
            </div>
          </>
        )}

        {activeTab === 'capture' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
            {/* Capture/Upload Section */}
            <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">Capture or Upload ECG</h2>
              <p className="text-gray-600 mb-6">Start by capturing a photo or uploading an existing ECG image</p>

              {/* Camera Preview / Uploaded Image */}
              <div className="border-2 border-dashed border-teal-300 rounded-xl p-4 sm:p-6 mb-6 bg-gray-50">
                {uploadedImage ? (
                  <div className="relative">
                    <img 
                      src={uploadedImage} 
                      alt="Uploaded ECG" 
                      className="w-full h-auto rounded-lg"
                    />
                    <button
                      onClick={() => setUploadedImage(null)}
                      className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-teal-400 rounded-lg p-8 sm:p-16 flex flex-col items-center justify-center">
                    <Camera className="w-16 h-16 text-gray-400 mb-4" />
                    <p className="text-gray-700 font-medium mb-1">Camera Preview</p>
                    <p className="text-gray-500 text-sm">Position your ECG paper here</p>
                  </div>
                )}
              </div>

              {/* Instruction */}
              {!uploadedImage && (
                <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 mb-6 flex items-start space-x-3">
                  <Info className="w-5 h-5 text-teal-600 mt-0.5 flex-shrink-0" />
                  <p className="text-teal-800 text-sm">Align the ECG paper horizontally within the frame</p>
                </div>
              )}

              {/* Action Buttons */}
              {!uploadedImage ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <label className="bg-teal-500 hover:bg-teal-600 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 cursor-pointer transition-colors">
                    <Camera className="w-5 h-5" />
                    <span>Capture Photo</span>
                    <input 
                      type="file" 
                      accept="image/*" 
                      capture="environment"
                      className="hidden" 
                      onChange={handleCameraCapture}
                    />
                  </label>
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
              ) : (
                <div className="space-y-3">
                  <button className="w-full bg-teal-500 hover:bg-teal-600 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 shadow-md hover:shadow-lg transition-all">
                    <Play className="w-5 h-5" />
                    <span>Run Analysis</span>
                  </button>
                  <div className="grid grid-cols-2 gap-3">
                    <label className="bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-4 py-2 rounded-lg font-medium flex items-center justify-center space-x-2 text-sm cursor-pointer transition-colors">
                      <Camera className="w-4 h-4" />
                      <span>Recapture</span>
                      <input 
                        type="file" 
                        accept="image/*" 
                        capture="environment"
                        className="hidden" 
                        onChange={handleCameraCapture}
                      />
                    </label>
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
              )}
            </div>

            {/* Guidelines Section */}
            <div className="bg-white rounded-xl p-6 sm:p-8 shadow-sm border border-gray-200">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">Capture Guidelines</h2>
              <p className="text-gray-600 mb-6">Follow these best practices for optimal results</p>

              <div className="space-y-5">
                {/* Guideline 1 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Place ECG paper on a flat surface</p>
                    <p className="text-sm text-gray-600">Avoid wrinkles or folds in the paper</p>
                  </div>
                </div>

                {/* Guideline 2 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Capture image in landscape orientation</p>
                    <p className="text-sm text-gray-600">Horizontal format works best</p>
                  </div>
                </div>

                {/* Guideline 3 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Ensure the full ECG sheet is visible</p>
                    <p className="text-sm text-gray-600">Include all 12 leads in the frame</p>
                  </div>
                </div>

                {/* Guideline 4 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Keep the camera parallel to the paper</p>
                    <p className="text-sm text-gray-600">Avoid angled or tilted shots</p>
                  </div>
                </div>

                {/* Guideline 5 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Avoid strong shadows and glare</p>
                    <p className="text-sm text-gray-600">Use diffused lighting when possible</p>
                  </div>
                </div>

                {/* Guideline 6 */}
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-teal-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Ensure good lighting</p>
                    <p className="text-sm text-gray-600">Natural light or bright indoor lighting</p>
                  </div>
                </div>
              </div>

              {/* Pro Tip */}
              <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start space-x-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-blue-900 mb-1">Pro Tip</p>
                  <p className="text-sm text-blue-800">For best results, clean the ECG paper surface and ensure there are no obstructions or annotations that might interfere with the analysis.</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Navigation Toggle (for demo) */}
      <div className="fixed bottom-6 right-6">
        <button
          onClick={() => setActiveTab(activeTab === 'home' ? 'capture' : 'home')}
          className="bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-full shadow-lg text-sm font-medium"
        >
          {activeTab === 'home' ? 'View Capture Page' : 'Back to Home'}
        </button>
      </div>
    </div>
  );
}