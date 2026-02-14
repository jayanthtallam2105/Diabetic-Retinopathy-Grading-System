import React from 'react';
import ResultCard from './ResultCard.jsx';

const MODEL_META = [
  { key: 'efficientnet', name: 'EfficientNet-B0' },
  { key: 'resnet50', name: 'ResNet50' },
  { key: 'vit', name: 'Vision Transformer (ViT-B/16)' },
  { key: 'hybrid_effvit', name: 'Hybrid EfficientNet + ViT' },
  { key: 'hybrid_resvit', name: 'Hybrid ResNet50 + ViT' },
];

const Dashboard = ({ results, quality }) => {
  if (!results) {
    return (
      <div className="mt-6 text-sm text-slate-500">
        Upload a retinal image and click <span className="font-semibold">Predict</span>{' '}
        to view the comparative model outputs.
      </div>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-50">
          Model Comparison Dashboard
        </h2>
        <p className="text-xs text-slate-500">
          All five models are executed on the same image for direct comparison.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {MODEL_META.map(({ key, name }) => (
          <ResultCard
            key={key}
            modelKey={key}
            modelName={name}
            result={results[key]}
          />
        ))}
      </div>
    </div>
  );
};

export default Dashboard;

