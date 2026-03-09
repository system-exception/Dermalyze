
import React from 'react';

interface HelpScreenProps {
  onBack: () => void;
}

const HelpScreen: React.FC<HelpScreenProps> = ({ onBack }) => {
  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 min-h-screen">
      <main className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
        <div className="max-w-2xl w-full">
          <div className="bg-white rounded-3xl border border-slate-200 p-8 sm:p-12 shadow-sm">
            <header className="mb-10 text-center">
              <h1 className="text-3xl font-bold tracking-tight text-slate-900 mb-2">How Dermalyze Works</h1>
              <div className="w-12 h-1 bg-teal-600 mx-auto rounded-full"></div>
            </header>

            <div className="space-y-12">
              {/* Section 1: Step-by-Step Usage Guide */}
              <section className="space-y-4">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Step-by-Step Usage Guide</h2>
                <ol className="space-y-3 list-decimal list-inside text-sm text-slate-600 leading-relaxed">
                  <li>Upload a high-quality dermatoscopic image of the lesion.</li>
                  <li>The system preprocesses the image on the secure server.</li>
                  <li>Deep learning algorithms analyze lesion features.</li>
                  <li>A probability distribution across clinical classes is generated.</li>
                  <li>Review results as a supplementary diagnostic data point.</li>
                </ol>
              </section>

              {/* Section 2: System Flow Explanation */}
              <section className="space-y-4">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">System Workflow</h2>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                  <p className="text-xs font-semibold text-slate-500 tracking-tight">
                    Image Upload → Clinical Preprocessing → Model Inference → Probability Output → Decision Support
                  </p>
                </div>
              </section>

              {/* Section 3: Frequently Asked Questions */}
              <section className="space-y-6">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Frequently Asked Questions</h2>
                <div className="space-y-6">
                  <div className="space-y-1">
                    <p className="text-sm font-bold text-slate-800">Q: “Is this a standalone diagnostic tool?”</p>
                    <p className="text-sm text-slate-600">A: “No. Dermalyze is a decision support system designed to assist, not replace, the clinician's primary diagnosis.”</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-bold text-slate-800">Q: “What types of images are required?”</p>
                    <p className="text-sm text-slate-600">A: “Standard dermatoscopic RGB images of skin lesions are required for accurate analysis.”</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-bold text-slate-800">Q: “How should the results be interpreted?”</p>
                    <p className="text-sm text-slate-600">A: “Results indicate model confidence and should be evaluated alongside clinical history and physical examination.”</p>
                  </div>
                </div>
              </section>
            </div>

            <div className="mt-12 flex justify-center">
              <button 
                onClick={onBack}
                className="text-sm font-bold text-teal-600 hover:text-teal-700 transition-colors flex items-center gap-2 group"
              >
                <svg className="w-4 h-4 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Dashboard
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="py-8 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Clinical Support & Diagnostic Aid Suite
        </p>
      </footer>
    </div>
  );
};

export default HelpScreen;
