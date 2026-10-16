import React, { useState } from 'react';
import { Camera, CheckCircle2, ShieldAlert, Sparkles, Sliders, Layers } from 'lucide-react';
import '@google/model-viewer';

export const CustomerApp: React.FC = () => {
  const [isAdult, setIsAdult] = useState(false);
  const [optInKeepFrontPhoto, setOptInKeepFrontPhoto] = useState(false);
  const [activeTab, setActiveTab] = useState<'twin' | 'capture' | 'measurements'>('twin');

  // Sample client-side measurements conforming to MeasurementZone in fitcore
  const measurements = [
    { zone: 'Chest', value: '102.5 cm', imperial: '40.4 in', conf: '96%' },
    { zone: 'Waist', value: '84.0 cm', imperial: '33.1 in', conf: '94%' },
    { zone: 'Hip', value: '98.0 cm', imperial: '38.6 in', conf: '92%' },
    { zone: 'Shoulder Width', value: '46.0 cm', imperial: '18.1 in', conf: '90%' },
    { zone: 'Sleeve Length', value: '64.5 cm', imperial: '25.4 in', conf: '88%' },
    { zone: 'Torso Length', value: '52.0 cm', imperial: '20.5 in', conf: '89%' },
    { zone: 'Inseam', value: '81.5 cm', imperial: '32.1 in', conf: '95%' },
    { zone: 'Thigh', value: '58.0 cm', imperial: '22.8 in', conf: '91%' },
    { zone: 'Neck', value: '39.0 cm', imperial: '15.4 in', conf: '93%' },
    { zone: 'Garment Length', value: '72.0 cm', imperial: '28.3 in', conf: '95%' },
    { zone: 'Back Rise', value: '38.0 cm', imperial: '15.0 in', conf: '87%' },
    { zone: 'Upper Back', value: '44.5 cm', imperial: '17.5 in', conf: '91%' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-sky-500/10 via-indigo-500/10 to-transparent border border-sky-200 dark:border-sky-900/60">
        <div>
          <h1 className="text-2xl font-black text-slate-900 dark:text-white flex items-center space-x-2">
            <span>Customer 3D Twin &amp; Fit Portal</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-sky-500 text-white font-semibold">PWA</span>
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Deterministic biometric twin derived from 4 orthogonal photos.
          </p>
        </div>

        {/* Privacy summary badge */}
        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-300 shadow-sm">
          <ShieldAlert className="w-4 h-4 text-emerald-500" />
          <span>Zero-Retention Raw Photos &bull; Owner-Only Firestore Access</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-slate-200 dark:border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('twin')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === 'twin'
              ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>3D Twin Viewer</span>
        </button>
        <button
          onClick={() => setActiveTab('capture')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === 'capture'
              ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Camera className="w-4 h-4" />
          <span>4-Photo Capture</span>
        </button>
        <button
          onClick={() => setActiveTab('measurements')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === 'measurements'
              ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Sliders className="w-4 h-4" />
          <span>My Measurements</span>
        </button>
      </div>

      {/* Tab 1: 3D Twin Viewer */}
      {activeTab === 'twin' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 rounded-2xl glass-panel border border-slate-200 dark:border-slate-800 p-6 flex flex-col items-center justify-center min-h-[460px] relative overflow-hidden">
            <div className="absolute top-4 left-4 z-10 flex items-center space-x-2 bg-slate-900/80 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-lg">
              <Sparkles className="w-3.5 h-3.5 text-sky-400" />
              <span>@google/model-viewer 3D Canvas</span>
            </div>

            {/* Simulated 3D body twin representation */}
            <div className="w-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-sky-300 dark:border-sky-800 rounded-xl bg-sky-50/50 dark:bg-slate-900/50 p-6 text-center">
              <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-600 text-white flex items-center justify-center mb-4 shadow-lg shadow-sky-500/30">
                <Layers className="w-10 h-10 animate-bounce" />
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-lg">SMPL 3D Body Twin Loaded</h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mt-1">
                Interacting with neutral SMPL avatar. 6,890 vertices fitted from 4 orthogonal MediaPipe poses.
              </p>
              <div className="mt-4 flex gap-2">
                <span className="text-[11px] px-2.5 py-1 rounded-md bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 font-mono">
                  Scale: 178.5 cm
                </span>
                <span className="text-[11px] px-2.5 py-1 rounded-md bg-indigo-100 dark:bg-indigo-950/60 text-indigo-800 dark:text-indigo-300 font-mono">
                  SMPL Beta: [10 parameters]
                </span>
              </div>
            </div>
          </div>

          {/* Quick Stats Panel */}
          <div className="space-y-4">
            <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-3">
                Key Fitted Zones
              </h3>
              <div className="space-y-2.5">
                {measurements.slice(0, 5).map((m) => (
                  <div key={m.zone} className="flex justify-between items-center text-sm">
                    <span className="text-slate-600 dark:text-slate-400">{m.zone}</span>
                    <span className="font-semibold text-slate-900 dark:text-white">{m.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-3">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-sky-500" />
                <span>Language &amp; Framing Rule</span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                All fit assessments strictly describe garment ease and dimensions. We never evaluate, categorize, or judge your body shape or weight.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: 4-Photo Capture */}
      {activeTab === 'capture' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">4-Photo Guided Capture</h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Capture 4 orthogonal views: Front, Back, Left Profile, and Right Profile.
            </p>
          </div>

          {/* Consent Checkboxes */}
          <div className="p-4 rounded-xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60 space-y-3">
            <label className="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={isAdult}
                onChange={(e) => setIsAdult(e.target.checked)}
                className="mt-1 rounded text-sky-600 focus:ring-sky-500 h-4 w-4"
              />
              <span className="text-sm font-medium text-amber-900 dark:text-amber-200">
                I confirm that I am 18 years of age or older (Mandatory prior to body capture).
              </span>
            </label>

            <label className="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={optInKeepFrontPhoto}
                onChange={(e) => setOptInKeepFrontPhoto(e.target.checked)}
                className="mt-1 rounded text-sky-600 focus:ring-sky-500 h-4 w-4"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">
                (Optional) Opt-in: Keep only the front photo for virtual try-on previews. By default, raw body photos are automatically deleted upon geometry calculation.
              </span>
            </label>
          </div>

          {/* Photo slots */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {['Front View', 'Back View', 'Left Profile', 'Right Profile'].map((view) => (
              <div
                key={view}
                className="p-6 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 flex flex-col items-center justify-center space-y-2 hover:border-sky-500 transition-colors cursor-pointer bg-slate-50 dark:bg-slate-900/40"
              >
                <Camera className="w-8 h-8 text-slate-400" />
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">{view}</span>
                <span className="text-[10px] text-slate-400">Click to upload</span>
              </div>
            ))}
          </div>

          <button
            disabled={!isAdult}
            className={`w-full py-3 rounded-xl font-bold transition-all ${
              isAdult
                ? 'bg-sky-600 hover:bg-sky-500 text-white shadow-md shadow-sky-600/25'
                : 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
            }`}
          >
            {isAdult ? 'Process 3D Twin via Geometry Service' : 'Confirm 18+ to Proceed'}
          </button>
        </div>
      )}

      {/* Tab 3: All Measurements */}
      {activeTab === 'measurements' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Full Metric Anatomic Profile</h2>
            <span className="text-xs text-slate-500">Internal Storage: Centimeters (cm)</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {measurements.map((m) => (
              <div key={m.zone} className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800 flex justify-between items-center">
                <div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">{m.zone}</p>
                  <p className="text-base font-bold text-slate-900 dark:text-white">{m.value}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-400">{m.imperial}</p>
                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">{m.conf} confidence</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
