import React, { useState } from 'react';
import axios from 'axios';
import Upload from './components/Upload.jsx';
import Dashboard from './components/Dashboard.jsx';

// Backend FastAPI base URL
const API_BASE_URL = 'http://127.0.0.1:8000';

const App = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResults(null);
    setError(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please upload a retinal image before running prediction.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      // New response structure: { quality: {...}, predictions: {...} }
      setResults(response.data);
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        'Failed to retrieve predictions from the server.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-900 bg-slate-950/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold tracking-tight text-slate-50">
              Diabetic Retinopathy Grading
            </h1>
            <p className="text-xs md:text-sm text-slate-400 mt-1">
              Research prototype comparing EfficientNet, ResNet, ViT and hybrid CNN–Transformer
              models on retinal fundus images.
            </p>
          </div>
          <div className="hidden md:flex flex-col items-end text-[11px] text-slate-500">
            <span>Backend: FastAPI · PyTorch · timm</span>
            <span>Frontend: React · Vite · Tailwind · Axios</span>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-4 py-6 md:py-8">
          <Upload
            onFileSelect={handleFileSelect}
            previewUrl={previewUrl}
            disabled={loading}
          />

          <div className="mt-4 flex flex-col md:flex-row gap-4 items-start">
            <button
              type="button"
              onClick={handlePredict}
              disabled={loading || !file}
              className={`inline-flex items-center justify-center px-5 py-2.5 rounded-xl text-sm font-semibold transition
                ${
                  loading || !file
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-md shadow-emerald-500/30'
                }`}
            >
              {loading ? 'Running models…' : 'Predict with All Models'}
            </button>

            <div className="text-xs text-slate-500">
              The backend runs five deep learning models concurrently and returns the DR grade,
              confidence, and softmax probabilities for each of the 5 severity levels.
            </div>
          </div>

          {loading && (
            <div className="mt-6 flex items-center gap-3 text-sm text-slate-300">
              <div className="w-8 h-8 border-4 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              <span>Running EfficientNet, ResNet, ViT and hybrid models on the image…</span>
            </div>
          )}

          {error && (
            <div className="mt-5 rounded-xl border border-red-500/70 bg-red-950/70 px-4 py-3 text-sm text-red-100">
              <div className="font-semibold mb-1">Error</div>
              <div>{error}</div>
            </div>
          )}

          {!loading && results && (
            <>
              {results.quality && (
                <div className={`mt-5 rounded-xl border px-4 py-3 text-sm ${
                  results.quality.accepted
                    ? results.quality.image_quality_score >= 80
                      ? 'border-emerald-500/70 bg-emerald-950/30 text-emerald-100'
                      : results.quality.image_quality_score >= 60
                      ? 'border-yellow-500/70 bg-yellow-950/30 text-yellow-100'
                      : 'border-orange-500/70 bg-orange-950/30 text-orange-100'
                    : 'border-red-500/70 bg-red-950/70 text-red-100'
                }`}>
                  <div className="font-semibold mb-1">
                    Image Quality: {results.quality.accepted 
                      ? (results.quality.image_quality_score >= 80 ? 'Good' 
                         : results.quality.image_quality_score >= 60 ? 'Fair' : 'Poor')
                      : 'Rejected'}
                  </div>
                  <div className="text-xs opacity-90">{results.quality.message}</div>
                  {results.quality.accepted && (
                    <div className="mt-2 text-xs opacity-75">
                      Blur: {results.quality.blur_score.toFixed(1)} | 
                      Brightness: {results.quality.brightness_score.toFixed(1)}% | 
                      Retina detected: {results.quality.retina_detected ? 'Yes' : 'No'}
                    </div>
                  )}
                </div>
              )}
              <Dashboard results={results.predictions || results} quality={results.quality} />
            </>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-900 bg-slate-950">
        <div className="max-w-6xl mx-auto px-4 py-3 text-[11px] text-slate-500 flex flex-col md:flex-row justify-between gap-2">
          <span>Research prototype – not for clinical use.</span>
          <span>Grades: 0–No DR, 1–Mild, 2–Moderate, 3–Severe, 4–Proliferative.</span>
        </div>
      </footer>
    </div>
  );
};

export default App;

