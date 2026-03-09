
import React from 'react';

interface AboutScreenProps {
  onBack: () => void;
}

const AboutScreen: React.FC<AboutScreenProps> = ({ onBack }) => {
  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 min-h-screen">
      <main className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
        <div className="max-w-2xl w-full">
          <div className="bg-white rounded-3xl border border-slate-200 p-8 sm:p-12 shadow-sm">
            <header className="mb-10 text-center">
              <h1 className="text-3xl font-bold tracking-tight text-slate-900 mb-2">About Dermalyze</h1>
              <div className="w-12 h-1 bg-teal-600 mx-auto rounded-full"></div>
            </header>

            <div className="space-y-10">
              {/* Section 1: Overview */}
              <section className="space-y-3">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Application Overview</h2>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Dermalyze is a clinical decision support application designed to assist dermatologists and healthcare professionals in analyzing dermatoscopic skin lesion images. 
                  The system performs multi-class classification and returns probabilistic outputs to aid diagnostic assessment.
                </p>
              </section>

              {/* Section 2: Dataset */}
              <section className="space-y-3">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Dataset Used</h2>
                <p className="text-sm text-slate-600 leading-relaxed">
                  The system is trained and evaluated using the HAM10000 (Human Against Machine with 10000 training images) dataset. 
                  HAM10000 is a standard dataset containing seven common skin lesion categories used for validating diagnostic support systems.
                </p>
              </section>

              {/* Section 3: Disclaimer */}
              <section className="space-y-3 p-6 bg-red-50/50 rounded-2xl border border-red-100">
                <h2 className="text-xs font-bold text-red-600 uppercase tracking-widest">Clinical Disclaimer</h2>
                <p className="text-sm text-slate-600 leading-relaxed italic">
                  “Dermalyze is a supportive diagnostic aid. It does not provide definitive medical diagnosis and should be used in conjunction with professional clinical evaluation. 
                  The final diagnostic decision remains the responsibility of the attending physician.”
                </p>
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

export default AboutScreen;
