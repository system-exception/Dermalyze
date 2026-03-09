import React from 'react';
import {
  ClassInfo,
  getRiskSeverity,
  getRiskBadgeStyles,
  getRiskLabel,
  getConfidenceColor,
} from '../../lib/classInfo';

interface ResultCardProps {
  classId: string;
  className: string;
  confidence: number;
  info: ClassInfo;
  imageUrl?: string | null;
}

/**
 * ResultCard — Primary focal point of the analysis screen.
 *
 * UX Laws applied:
 * - Visual Hierarchy: largest type size for the predicted class, establishing
 *   a clear F-pattern reading flow (diagnosis → confidence → risk).
 * - Aesthetic-Usability Effect: clean card with generous whitespace feels
 *   trustworthy and professional — critical in a clinical context.
 * - Fitts's Law: risk badge is large enough to be instantly perceived.
 * - Law of Proximity: image placed alongside diagnosis for immediate
 *   visual correlation between the lesion and the classification.
 */
const ResultCard: React.FC<ResultCardProps> = ({ classId, className, confidence, info, imageUrl }) => {
  const severity = getRiskSeverity(info.riskLevel);
  const badge = getRiskBadgeStyles(severity);
  const confidenceColor = getConfidenceColor(confidence);

  return (
    <section className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Top accent strip keyed to risk severity */}
      <div
        className={`h-1 w-full ${
          severity === 'critical' || severity === 'high'
            ? 'bg-red-500'
            : severity === 'moderate'
            ? 'bg-amber-400'
            : 'bg-emerald-500'
        }`}
      />

      <div className="px-4 sm:px-8 pt-5 sm:pt-7 pb-6 sm:pb-8">
        <div className="flex flex-col md:flex-row md:items-start gap-6">
          {/* Left — Diagnosis info */}
          <div className="flex-1 min-w-0">
            {/* Overline label — small, muted to set context */}
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-4">
              Predicted Classification
            </p>

            {/* Primary diagnosis — largest element on screen */}
            <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight mb-1">
              {className}
            </h2>
            <p className="text-sm text-slate-400 font-medium mb-6">
              Class: <span className="font-semibold text-slate-500 uppercase">{classId}</span>
            </p>

            {/* Confidence + Risk — grouped via Law of Proximity */}
            <div className="flex flex-wrap items-center gap-4">
              {/* Confidence score */}
              <div className="flex items-baseline gap-2">
                <span className={`text-4xl font-extrabold tabular-nums ${confidenceColor}`}>
                  {confidence.toFixed(1)}
                  <span className="text-2xl">%</span>
                </span>
                <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                  Confidence
                </span>
              </div>

              {/* Vertical divider */}
              <div className="hidden sm:block w-px h-8 bg-slate-200" />

              {/* Risk badge */}
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${badge.bg} ${badge.text} ${badge.border}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`} />
                {getRiskLabel(severity)}
              </span>
            </div>
          </div>

          {/* Right — Analyzed image */}
          <div className="w-full md:w-48 lg:w-56 flex-shrink-0">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-3 md:text-right">
              Analyzed Image
            </p>
            <div className="aspect-square w-full bg-slate-50 rounded-lg overflow-hidden border border-slate-100 flex items-center justify-center">
              {imageUrl ? (
                <img
                  src={imageUrl}
                  alt="Analyzed lesion"
                  className="w-full h-full object-cover"
                  draggable={false}
                />
              ) : (
                <div className="text-slate-300 flex flex-col items-center gap-2">
                  <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <span className="text-xs">No image</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResultCard;
