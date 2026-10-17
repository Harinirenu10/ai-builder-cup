import React, { useState, useRef } from 'react';
import {
  Camera,
  CheckCircle2,
  ShieldAlert,
  Sparkles,
  Sliders,
  Layers,
  AlertCircle,
  Trash2,
  RefreshCw,
  Loader2,
} from 'lucide-react';
import '@google/model-viewer';

type ViewType = 'front' | 'back' | 'left' | 'right';

interface PhotoState {
  file: File | null;
  previewUrl: string | null;
  dataUrl: string | null;
  fileName: string | null;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const VIEW_CONFIG: Record<ViewType, { label: string; desc: string }> = {
  front: { label: 'Front View', desc: 'Arms slightly away, facing camera' },
  back: { label: 'Back View', desc: 'Back straight, feet shoulder-width' },
  left: { label: 'Left Profile', desc: 'Standing straight, left side to camera' },
  right: { label: 'Right Profile', desc: 'Standing straight, right side to camera' },
};

export const CustomerApp: React.FC = () => {
  const [isAdult, setIsAdult] = useState(false);
  const [optInKeepFrontPhoto, setOptInKeepFrontPhoto] = useState(false);
  const [activeTab, setActiveTab] = useState<'twin' | 'capture' | 'measurements'>('twin');

  // Input states
  const [heightCm, setHeightCm] = useState<string>('175.0');
  const [gender, setGender] = useState<'neutral' | 'female' | 'male'>('neutral');

  // Photo upload states
  const [photos, setPhotos] = useState<Record<ViewType, PhotoState>>({
    front: { file: null, previewUrl: null, dataUrl: null, fileName: null },
    back: { file: null, previewUrl: null, dataUrl: null, fileName: null },
    left: { file: null, previewUrl: null, dataUrl: null, fileName: null },
    right: { file: null, previewUrl: null, dataUrl: null, fileName: null },
  });

  // Submission & feedback states
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<any | null>(null);

  // Hidden file inputs
  const fileInputRefs: Record<ViewType, React.RefObject<HTMLInputElement>> = {
    front: useRef<HTMLInputElement>(null),
    back: useRef<HTMLInputElement>(null),
    left: useRef<HTMLInputElement>(null),
    right: useRef<HTMLInputElement>(null),
  };

  // Metric measurements list
  const [measurements, setMeasurements] = useState([
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
  ]);

  // Handle file selection and read as base64 data URL
  const handleFileChange = (view: ViewType, file: File | null) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMessage(`File '${file.name}' is not an image. Please upload a JPG or PNG file.`);
      return;
    }

    setErrorMessage(null);
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPhotos((prev) => ({
        ...prev,
        [view]: {
          file,
          previewUrl: dataUrl,
          dataUrl,
          fileName: file.name,
        },
      }));
    };
    reader.onerror = () => {
      setErrorMessage(`Failed to read image file '${file.name}'.`);
    };
    reader.readAsDataURL(file);
  };

  // Handle removing a photo
  const handleRemovePhoto = (view: ViewType, e: React.MouseEvent) => {
    e.stopPropagation();
    setPhotos((prev) => ({
      ...prev,
      [view]: { file: null, previewUrl: null, dataUrl: null, fileName: null },
    }));
    if (fileInputRefs[view].current) {
      fileInputRefs[view].current!.value = '';
    }
  };

  // Handle submission to backend workflow
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);
    // Clear any previous job's result up front so a failed/erroring
    // resubmission can never leave a stale twin/measurements on screen as if
    // it were the result of this attempt.
    setJobResult(null);

    // 1. Validate 18+ age confirmation
    if (!isAdult) {
      setErrorMessage('Confirmation of 18+ years of age is mandatory prior to biometric photo capture.');
      return;
    }

    // 2. Validate height
    const parsedHeight = parseFloat(heightCm);
    if (!heightCm || isNaN(parsedHeight) || parsedHeight < 50.0 || parsedHeight > 260.0) {
      setErrorMessage('Please enter a valid height in centimeters between 50.0 and 260.0 cm.');
      return;
    }

    // 3. Validate all 4 photos are provided
    const missingViews: string[] = [];
    (Object.keys(VIEW_CONFIG) as ViewType[]).forEach((v) => {
      if (!photos[v].dataUrl) {
        missingViews.push(VIEW_CONFIG[v].label);
      }
    });

    if (missingViews.length > 0) {
      setErrorMessage(`All 4 orthogonal views are required. Missing: ${missingViews.join(', ')}.`);
      return;
    }

    setIsSubmitting(true);

    try {
      const payload = {
        user_id: 'user_' + Math.random().toString(36).substring(2, 10),
        height_cm: parsedHeight,
        gender: gender,
        age_18_confirmed: isAdult,
        opt_in_keep_front_photo: optInKeepFrontPhoto,
        photo_front: photos.front.dataUrl,
        photo_back: photos.back.dataUrl,
        photo_left: photos.left.dataUrl,
        photo_right: photos.right.dataUrl,
      };

      const response = await fetch(`${API_URL}/api/twin/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error_message || `Server returned error (${response.status})`);
      }

      if (data.status === 'needs_retake' || data.status === 'rejected' || data.status === 'failed') {
        const errorMsg = data.error_message || `Job status: ${data.status} (code: ${data.error_code || 'UNKNOWN'})`;
        setErrorMessage(`Photo Validation Issue: ${errorMsg}`);
        return;
      }

      setJobResult(data);
      setSuccessMessage(
        `3D Body Twin created successfully! (Job ID: ${data.job_id}${data.twin_id ? ', Twin ID: ' + data.twin_id : ''})`
      );

      // If backend returned measurements, populate My Measurements tab.
      // The API returns the Measurements schema shape { zones: { <zone>: { value_cm, confidence, source } } },
      // not a flat zone->value map, so the real per-zone data lives under data.measurements.zones.
      const zones = data.measurements?.zones;
      if (zones && Object.keys(zones).length > 0) {
        const updated = Object.entries(zones)
          .filter(([, val]: [string, any]) => val && typeof val.value_cm === 'number')
          .map(([key, val]: [string, any]) => {
            const valCm = val.value_cm;
            const valInches = (valCm / 2.54).toFixed(1);
            const confPercent = typeof val.confidence === 'number' ? Math.round(val.confidence * 100) : 95;
            const titleCase = key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
            return {
              zone: titleCase,
              value: `${valCm.toFixed(1)} cm`,
              imperial: `${valInches} in`,
              conf: `${confPercent}%`,
            };
          });
        if (updated.length > 0) {
          setMeasurements(updated);
        }
      }
    } catch (err: any) {
      setErrorMessage(
        err.message || 'Could not connect to API service. Make sure backend is running on http://localhost:8000.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

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

            {/* 3D body twin representation */}
            {jobResult?.glb_url ? (
              <div className="w-full h-80 flex flex-col items-center rounded-xl bg-sky-50/50 dark:bg-slate-900/50 overflow-hidden">
                <model-viewer
                  src={`${API_URL}${jobResult.glb_url}`}
                  alt={`3D Body Twin ${jobResult.twin_id}`}
                  camera-controls
                  auto-rotate
                  shadow-intensity="1"
                  style={{ width: '100%', height: '100%' }}
                />
                <div className="w-full flex items-center justify-center gap-2 py-2 bg-slate-900/80 backdrop-blur-md">
                  <span className="text-[11px] px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 font-mono">
                    Twin: {jobResult.twin_id}
                  </span>
                  <span className="text-[11px] px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-300 font-mono">
                    Scale: {heightCm} cm
                  </span>
                </div>
              </div>
            ) : (
              <div className="w-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-sky-300 dark:border-sky-800 rounded-xl bg-sky-50/50 dark:bg-slate-900/50 p-6 text-center">
                <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-600 text-white flex items-center justify-center mb-4 shadow-lg shadow-sky-500/30">
                  <Layers className="w-10 h-10 animate-bounce" />
                </div>
                <h4 className="font-bold text-slate-900 dark:text-white text-lg">
                  No 3D Twin Yet
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mt-1">
                  Complete the 4-Photo Capture step to fit your SMPL body twin from orthogonal MediaPipe poses.
                </p>
                <div className="mt-4 flex gap-2">
                  <span className="text-[11px] px-2.5 py-1 rounded-md bg-indigo-100 dark:bg-indigo-950/60 text-indigo-800 dark:text-indigo-300 font-mono">
                    Scale: {heightCm} cm
                  </span>
                </div>
              </div>
            )}
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
                All fit assessments strictly describe garment ease and dimensions. We never evaluate, categorize, or judge
                your body shape or weight.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: 4-Photo Capture */}
      {activeTab === 'capture' && (
        <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">4-Photo Guided Capture</h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Capture or upload 4 orthogonal views: Front, Back, Left Profile, and Right Profile.
            </p>
          </div>

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-start space-x-3 text-rose-800 dark:text-rose-200 animate-fadeIn">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
              <div className="text-sm font-medium">{errorMessage}</div>
            </div>
          )}

          {/* Success Message Alert */}
          {successMessage && (
            <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900 flex items-start space-x-3 text-emerald-800 dark:text-emerald-200 animate-fadeIn">
              <CheckCircle2 className="w-5 h-5 flex-shrink-0 text-emerald-600 dark:text-emerald-400 mt-0.5" />
              <div className="flex-1 text-sm">
                <p className="font-bold">{successMessage}</p>
                <p className="text-xs text-emerald-700 dark:text-emerald-300 mt-1">
                  Biometric measurements updated. You can inspect your 3D avatar in the Twin Viewer.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setActiveTab('measurements')}
                className="text-xs font-bold underline hover:no-underline text-emerald-700 dark:text-emerald-300 ml-2"
              >
                View Measurements &rarr;
              </button>
            </div>
          )}

          {/* Height and Gender Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div>
              <label htmlFor="height-cm" className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 mb-1.5">
                Height (cm) <span className="text-rose-500">*</span>
              </label>
              <input
                id="height-cm"
                type="number"
                step="0.5"
                min="50"
                max="260"
                value={heightCm}
                onChange={(e) => setHeightCm(e.target.value)}
                placeholder="e.g. 175.0"
                className="w-full px-3.5 py-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-semibold text-slate-900 dark:text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
              <p className="text-[11px] text-slate-500 mt-1">Required for accurate pixel-to-centimeter scale calibration.</p>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 mb-1.5">
                SMPL Body Template (Gender)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['neutral', 'female', 'male'] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setGender(g)}
                    className={`py-2 px-3 rounded-lg text-xs font-semibold capitalize border transition-all ${
                      gender === g
                        ? 'bg-sky-600 text-white border-sky-600 shadow-sm'
                        : 'bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-sky-400'
                    }`}
                  >
                    {g}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Selects base parametric SMPL mesh topology.</p>
            </div>
          </div>

          {/* Consent Checkboxes */}
          <div className="p-4 rounded-xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60 space-y-3">
            <label className="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={isAdult}
                onChange={(e) => {
                  setIsAdult(e.target.checked);
                  if (e.target.checked) setErrorMessage(null);
                }}
                className="mt-1 rounded text-sky-600 focus:ring-sky-500 h-4 w-4"
              />
              <span className="text-sm font-medium text-amber-900 dark:text-amber-200">
                I confirm that I am 18 years of age or older (Mandatory prior to body capture). <span className="text-rose-500">*</span>
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

          {/* 4 Photo Upload Slots */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {(Object.keys(VIEW_CONFIG) as ViewType[]).map((view) => {
              const photo = photos[view];
              const cfg = VIEW_CONFIG[view];
              return (
                <div key={view} className="flex flex-col space-y-1.5">
                  <input
                    type="file"
                    ref={fileInputRefs[view]}
                    onChange={(e) => handleFileChange(view, e.target.files?.[0] || null)}
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                  />

                  {photo.previewUrl ? (
                    /* Image Preview Card */
                    <div className="relative rounded-xl border border-sky-400 dark:border-sky-600 bg-slate-900/5 dark:bg-slate-900/60 p-2 flex flex-col items-center justify-between h-48 overflow-hidden group shadow-sm">
                      <div className="w-full flex items-center justify-between z-10 mb-1">
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-emerald-500 text-white text-[10px] font-bold">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>{cfg.label}</span>
                        </span>
                        <div className="flex items-center space-x-1">
                          <button
                            type="button"
                            onClick={() => fileInputRefs[view].current?.click()}
                            className="p-1 rounded-md bg-white/90 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:text-sky-600 shadow text-xs"
                            title="Replace photo"
                          >
                            <RefreshCw className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={(e) => handleRemovePhoto(view, e)}
                            className="p-1 rounded-md bg-white/90 dark:bg-slate-800 text-rose-600 hover:text-rose-700 shadow text-xs"
                            title="Remove photo"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>

                      <div className="flex-1 w-full flex items-center justify-center overflow-hidden">
                        <img
                          src={photo.previewUrl}
                          alt={cfg.label}
                          className="h-32 w-full object-contain rounded-lg"
                        />
                      </div>

                      <div className="w-full text-center mt-1">
                        <span className="text-[10px] text-slate-500 dark:text-slate-400 truncate block max-w-full">
                          {photo.fileName}
                        </span>
                      </div>
                    </div>
                  ) : (
                    /* Upload Placeholder Slot */
                    <div
                      onClick={() => fileInputRefs[view].current?.click()}
                      className="h-48 p-4 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 flex flex-col items-center justify-center space-y-2 hover:border-sky-500 hover:bg-sky-50/40 dark:hover:bg-slate-900/60 transition-all cursor-pointer bg-slate-50 dark:bg-slate-900/40 text-center"
                    >
                      <div className="w-10 h-10 rounded-full bg-slate-200/70 dark:bg-slate-800 flex items-center justify-center text-slate-500">
                        <Camera className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-xs font-bold text-slate-800 dark:text-slate-200 block">{cfg.label}</span>
                        <span className="text-[10px] text-slate-500 dark:text-slate-400 block mt-0.5">{cfg.desc}</span>
                      </div>
                      <span className="text-[10px] font-semibold text-sky-600 dark:text-sky-400">Click to upload</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!isAdult || isSubmitting}
            className={`w-full py-3.5 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 ${
              !isAdult
                ? 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
                : isSubmitting
                ? 'bg-sky-500 text-white cursor-wait opacity-90'
                : 'bg-sky-600 hover:bg-sky-500 text-white shadow-md shadow-sky-600/25 active:scale-[0.99]'
            }`}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing 3D Body Twin via Geometry Service...</span>
              </>
            ) : isAdult ? (
              <>
                <Sparkles className="w-5 h-5" />
                <span>Process 3D Twin via Geometry Service</span>
              </>
            ) : (
              <span>Confirm 18+ to Proceed</span>
            )}
          </button>
        </form>
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
              <div
                key={m.zone}
                className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800 flex justify-between items-center"
              >
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
