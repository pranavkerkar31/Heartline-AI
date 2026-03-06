"use client";
import React from "react";
import { Zap, Shield, TrendingUp } from "lucide-react";

interface LandingPageProps {
  onGetStarted: () => void;
}

export default function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <>
      {/* Hero Section */}
      <div className="text-center mb-12 sm:mb-16">
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
          ECG Digitisation & AI Analysis
        </h1>
        <p className="text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto">
          Transform traditional ECG paper records into digital format with
          AI-powered analysis. Detect heart conditions quickly and accurately
          using advanced machine learning.
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {/* Fast Processing */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
          <div className="bg-teal-100 rounded-lg p-3 w-fit mb-4">
            <Zap className="w-6 h-6 text-teal-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Fast Processing
          </h3>
          <p className="text-gray-600">AI-powered analysis in seconds</p>
        </div>

        {/* Accurate Detection */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
          <div className="bg-blue-100 rounded-lg p-3 w-fit mb-4">
            <Shield className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Accurate Detection
          </h3>
          <p className="text-gray-600">CNN-based classification system</p>
        </div>

        {/* Advanced Analytics */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
          <div className="bg-purple-100 rounded-lg p-3 w-fit mb-4">
            <TrendingUp className="w-6 h-6 text-purple-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Advanced Analytics
          </h3>
          <p className="text-gray-600">Comprehensive heart health insights</p>
        </div>
      </div>

      {/* CTA Button */}
      <div className="text-center">
        <button
          onClick={onGetStarted}
          className="bg-teal-500 hover:bg-teal-600 text-white px-8 py-3 rounded-lg font-medium text-lg shadow-md hover:shadow-lg transition-all"
        >
          Get Started
        </button>
      </div>
    </>
  );
}