"use client";
import React, { useState } from "react";
import LandingPage from "@/components/LandingPage";
import CameraPage from "@/components/CameraPage";
import LiveFeed from "@/components/LiveFeed";

export default function ECGDigitisation() {
  const [activeTab, setActiveTab] = useState("home");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-teal-500 rounded-lg p-2">
                <svg
                  className="w-6 h-6 text-white"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
              <span className="text-xl font-semibold text-gray-900">
                Heartline AI
              </span>
            </div>
            
            <nav className="hidden md:flex items-center space-x-8">
              <button 
                onClick={() => setActiveTab("home")}
                className={`text-sm font-medium transition-colors ${activeTab === 'home' ? 'text-teal-600 border-b-2 border-teal-600' : 'text-gray-500 hover:text-gray-900'}`}
              >
                Dashboard
              </button>
              <button 
                onClick={() => setActiveTab("monitor")}
                className={`text-sm font-medium transition-colors ${activeTab === 'monitor' ? 'text-teal-600 border-b-2 border-teal-600' : 'text-gray-500 hover:text-gray-900'}`}
              >
                Live Monitor
              </button>
              <button 
                onClick={() => setActiveTab("capture")}
                className={`text-sm font-medium transition-colors ${activeTab === 'capture' ? 'text-teal-600 border-b-2 border-teal-600' : 'text-gray-500 hover:text-gray-900'}`}
              >
                Simulator
              </button>
            </nav>

            <div className="flex items-center space-x-4">
              <button className="text-gray-700 hover:text-gray-900 font-medium">
                Login
              </button>
              <button className="bg-teal-500 hover:bg-teal-600 text-white px-4 py-2 rounded-lg font-medium flex items-center space-x-2">
                <span>Sign Up</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {activeTab === "home" && (
          <LandingPage onGetStarted={() => setActiveTab("monitor")} />
        )}

        {activeTab === "monitor" && <LiveFeed />}

        {activeTab === "capture" && <CameraPage />}
      </main>

      {/* Navigation Toggle (for demo) */}
      <div className="fixed bottom-6 right-6 flex flex-col gap-3">
        <button
          onClick={() => setActiveTab("monitor")}
          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-full shadow-lg text-xs font-bold uppercase tracking-widest flex items-center gap-2"
        >
          <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
          Live Feed
        </button>
      </div>
    </div>
  );
}
