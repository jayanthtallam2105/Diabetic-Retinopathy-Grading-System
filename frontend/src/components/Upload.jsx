import React from 'react';

const Upload = ({ onFileSelect, previewUrl, disabled }) => {
  const handleChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 flex flex-col md:flex-row gap-6 items-center">
      <div className="flex-1">
        <h2 className="text-lg font-semibold text-slate-50 mb-2">
          Upload Retinal Fundus Image
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Select a color fundus photograph (JPG or PNG). The system will send it to the backend
          and run all five deep learning models in parallel.
        </p>
        <label className="inline-flex items-center px-4 py-2 rounded-lg cursor-pointer bg-emerald-500/90 hover:bg-emerald-400 text-slate-950 font-medium transition">
          <span>Choose Image</span>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleChange}
            disabled={disabled}
          />
        </label>
      </div>
      <div className="w-40 h-40 border border-slate-700 rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Retinal preview"
            className="w-full h-full object-cover"
          />
        ) : (
          <span className="text-xs text-slate-500 text-center px-3">
            Image preview will appear here after selection.
          </span>
        )}
      </div>
    </div>
  );
};

export default Upload;

