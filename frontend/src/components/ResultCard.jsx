import React, { useState, useRef, useEffect } from 'react';

const GRADE_LABELS = [
  '0 – No DR',
  '1 – Mild',
  '2 – Moderate',
  '3 – Severe',
  '4 – Proliferative',
];

const CLASS_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];

const getSeverityClasses = (grade) => {
  switch (grade) {
    case 0:
      return {
        badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/60',
        dot: 'bg-emerald-400',
      };
    case 1:
      return {
        badge: 'bg-lime-500/15 text-lime-300 border-lime-500/60',
        dot: 'bg-lime-400',
      };
    case 2:
      return {
        badge: 'bg-yellow-500/15 text-yellow-200 border-yellow-500/60',
        dot: 'bg-yellow-400',
      };
    case 3:
      return {
        badge: 'bg-orange-500/15 text-orange-200 border-orange-500/60',
        dot: 'bg-orange-400',
      };
    case 4:
      return {
        badge: 'bg-red-500/15 text-red-200 border-red-500/60',
        dot: 'bg-red-400',
      };
    default:
      return {
        badge: 'bg-slate-700/40 text-slate-200 border-slate-500/60',
        dot: 'bg-slate-400',
      };
  }
};

const ResultCard = ({ modelKey, modelName, result }) => {
  if (!result) return null;

  const hasError = result.error != null;
  const grade = result.grade;
  const confidence = result.confidence;
  const probs = result.probs || [];
  const gradcam = result.gradcam || null;

  const severity = getSeverityClasses(grade ?? -1);
  const confidencePct = confidence != null ? (confidence * 100).toFixed(1) : null;
  
  // Visualization state
  const [showGradcam, setShowGradcam] = useState(false);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' or 'heatmap'
  const [opacity, setOpacity] = useState(gradcam?.alpha || 0.6);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  // Draw bounding boxes on canvas overlay for CNN models
  useEffect(() => {
    if (!showGradcam || !gradcam || viewMode !== 'overlay' || !canvasRef.current || !imgRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const img = imgRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match image
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    // Draw bounding boxes if available (CNN models only)
    // Note: Backend already draws yellow boxes on overlay image
    // This canvas overlay is redundant but kept for compatibility
    const boxes = gradcam.bounding_boxes || [];
    if (boxes.length > 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#ffff00'; // Yellow (medical standard)
      ctx.lineWidth = 2;
      
      boxes.forEach((box) => {
        ctx.strokeRect(box.x, box.y, box.w, box.h);
      });
    }
  }, [showGradcam, gradcam, viewMode]);

  const getGradcamImageSrc = () => {
    if (!gradcam) return null;
    
    if (viewMode === 'heatmap') {
      // Use heatmap-only visualization (TURBO colormap)
      return `data:image/png;base64,${gradcam.heatmap}`;
    } else {
      // Use overlay (TURBO colormap, yellow bounding boxes already drawn on backend for CNNs)
      return `data:image/png;base64,${gradcam.overlay}`;
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg shadow-slate-950/60 flex flex-col gap-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-slate-50 tracking-tight">
            {modelName}
          </h3>
          <p className="text-[11px] text-slate-500 uppercase tracking-[0.16em] mt-0.5">
            {modelKey}
          </p>
        </div>

        {!hasError && grade != null && (
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-medium ${severity.badge}`}
          >
            <span className={`w-2 h-2 rounded-full ${severity.dot}`} />
            <span>{GRADE_LABELS[grade] ?? 'Unknown grade'}</span>
          </div>
        )}

        {hasError && (
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-red-500/70 bg-red-500/10 text-[11px] text-red-200">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            <span>Model error</span>
          </div>
        )}
      </div>

      {!hasError && grade != null && (
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col">
            <span className="text-xs text-slate-400">Predicted DR Grade</span>
            <span className="text-lg font-semibold text-slate-50">
              {grade} – {CLASS_LABELS[grade] ?? 'Unknown'}
            </span>
          </div>
          <div className="text-right">
            <span className="text-xs text-slate-400">Confidence</span>
            <div className="flex items-baseline gap-1">
              <span className="text-xl font-semibold text-emerald-400">
                {confidencePct}
              </span>
              <span className="text-xs text-slate-500">%</span>
            </div>
          </div>
        </div>
      )}

      {hasError && (
        <div className="text-xs text-red-200 bg-red-500/5 border border-red-500/40 rounded-lg px-3 py-2">
          {String(result.error)}
        </div>
      )}

      {!hasError && Array.isArray(probs) && probs.length === 5 && (
        <div className="mt-1 space-y-3">
          <div>
            <p className="text-xs font-medium text-slate-400 mb-2">
              Class Probabilities (Softmax)
            </p>
            <div className="space-y-1.5">
              {probs.map((p, idx) => {
                const pct = p * 100;
                const label = CLASS_LABELS[idx] ?? `Class ${idx}`;
                const isPred = idx === grade;
                return (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-20 text-[11px] text-slate-400">
                      {idx} – {label}
                    </div>
                    <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isPred ? 'bg-emerald-400' : 'bg-slate-500/80'
                        }`}
                        style={{ width: `${pct.toFixed(1)}%` }}
                      />
                    </div>
                    <div className="w-10 text-right text-[11px] text-slate-400">
                      {pct.toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Grad-CAM Controls */}
          {gradcam ? (
            <div className="space-y-2 border-t border-slate-800 pt-3">
              <div className="flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setShowGradcam((v) => !v)}
                  className={`px-3 py-1.5 rounded-lg text-[11px] font-medium border transition ${
                    showGradcam
                      ? 'bg-emerald-500/10 border-emerald-400 text-emerald-300'
                      : 'bg-slate-900/60 border-slate-700 text-slate-300 hover:border-emerald-400/70'
                  }`}
                >
                  {showGradcam ? 'Hide Grad-CAM' : 'Show Grad-CAM'}
                </button>
                <span className="text-[11px] text-slate-500">
                  Model attention areas
                </span>
              </div>

              {showGradcam && (
                <div className="space-y-2">
                  {/* View Mode Toggle - Raw removed */}
                  <div className="flex gap-1">
                    {['overlay', 'heatmap'].map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setViewMode(mode)}
                        className={`px-2 py-1 rounded text-[10px] font-medium capitalize transition ${
                          viewMode === mode
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/50'
                            : 'bg-slate-800 text-slate-400 border border-slate-700 hover:border-slate-600'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>

                  {/* Opacity Slider */}
                  {viewMode === 'overlay' && (
                    <div className="space-y-1">
                      <label className="text-[10px] text-slate-400 flex items-center justify-between">
                        <span>Overlay Opacity</span>
                        <span className="text-emerald-400">{Math.round(opacity * 100)}%</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={opacity}
                        onChange={(e) => setOpacity(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                      />
                    </div>
                  )}


                  {/* Visualization Display */}
                  <div className="relative w-full aspect-[4/3] overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
                    <img
                      ref={imgRef}
                      src={getGradcamImageSrc()}
                      alt={`${modelName} ${gradcam.method === 'attention_rollout' ? 'Attention' : 'Grad-CAM'}`}
                      className="w-full h-full object-contain"
                      style={
                        viewMode === 'overlay'
                          ? { opacity: opacity }
                          : {}
                      }
                      onLoad={() => {
                        // Trigger bounding box drawing after image loads
                        // Note: Backend already draws yellow boxes on overlay image
                        // This canvas overlay is redundant but kept for compatibility
                        if (gradcam.bounding_boxes && gradcam.bounding_boxes.length > 0) {
                          const canvas = canvasRef.current;
                          const img = imgRef.current;
                          if (canvas && img) {
                            canvas.width = img.naturalWidth || img.width;
                            canvas.height = img.naturalHeight || img.height;
                            const ctx = canvas.getContext('2d');
                            ctx.strokeStyle = '#ffff00'; // Yellow (medical standard)
                            ctx.lineWidth = 2;
                            gradcam.bounding_boxes.forEach((box) => {
                              ctx.strokeRect(box.x, box.y, box.w, box.h);
                            });
                          }
                        }
                      }}
                    />
                    {/* Canvas overlay for bounding boxes (CNN models only) */}
                    {gradcam.bounding_boxes && gradcam.bounding_boxes.length > 0 && (
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none"
                        style={{ mixBlendMode: 'normal' }}
                      />
                    )}
                  </div>
                  <p className="text-[10px] text-slate-500 text-center">
                    {gradcam.note || 'Highlighted regions indicate model attention — not clinical diagnosis.'}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-[11px] text-slate-500 text-center pt-2 border-t border-slate-800">
              Grad-CAM not available for this model.
            </div>
          )}
        </div>
      )}

      <div className="mt-1 pt-2 border-t border-slate-800/80 text-[11px] text-slate-500 flex justify-between">
        <span>Vision-only inference, 5-class DR grading</span>
        <span>Not for clinical decision-making</span>
      </div>
    </div>
  );
};

export default ResultCard;
