import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
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
  UserCheck,
  LogIn,
  Download,
  Share2,
  WifiOff,
  X,
  ExternalLink,
} from 'lucide-react';
import '@google/model-viewer';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, ApiError, loadObjectUrl } from '../lib/api';
import { useAuth } from '../lib/auth';
import { usePwaInstall, useOnlineStatus } from '../lib/pwa';
import { useCustomerProgress } from '../lib/useProgress';
import { useJobStatus } from '../lib/useJobStatus';

type ViewType = 'front' | 'back' | 'left' | 'right';

interface PhotoState {
  file: File | null;
  previewUrl: string | null;
  dataUrl: string | null;
  fileName: string | null;
}

const VIEW_CONFIG: Record<ViewType, { label: string; desc: string }> = {
  front: { label: 'Front View', desc: 'Arms raised out to shoulder height — clear gap at chest' },
  back: { label: 'Back View', desc: 'Back straight, feet shoulder-width' },
  left: { label: 'Left Profile', desc: 'Standing straight, left side to camera' },
  right: { label: 'Right Profile', desc: 'Standing straight, right side to camera' },
};

const parseZonesToMeasurements = (zones: Record<string, any>) => {
  return Object.entries(zones)
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
};

export const CustomerApp: React.FC = () => {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user, signedIn, signInWithDemo } = useAuth();

  // PWA and network connectivity helpers
  const { canInstall, promptInstall } = usePwaInstall();
  const isOnline = useOnlineStatus();

  // User-specific draft progress persistence
  const { progress, updateProgress } = useCustomerProgress();
  const progressLoadedRef = useRef(false);

  // Android Web Share Target parameter detection
  const sharedUrl = searchParams.get('url') || searchParams.get('text') || '';
  const sharedTitle = searchParams.get('title') || '';
  const [productUrlInput, setProductUrlInput] = useState<string>(() => sharedUrl);

  useEffect(() => {
    if (sharedUrl) {
      setProductUrlInput(sharedUrl);
    }
  }, [sharedUrl]);

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
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<any | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  // Live polling of in-flight twin fitting jobs
  const {
    job,
    status: jobStatus,
    stage: jobStage,
    progressPercent: jobProgressPercent,
    isTerminal: isJobTerminal,
  } = useJobStatus(activeJobId);

  // Hidden file inputs with environment camera capture for mobile
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

  // Synchronize persisted progress into local draft state
  useEffect(() => {
    if (progress && !progressLoadedRef.current) {
      if (progress.height_cm) setHeightCm(progress.height_cm);
      if (progress.gender) setGender(progress.gender);
      if (typeof progress.age_18_confirmed === 'boolean') setIsAdult(progress.age_18_confirmed);
      if (typeof progress.opt_in_keep_front_photo === 'boolean') setOptInKeepFrontPhoto(progress.opt_in_keep_front_photo);
      if (progress.active_tab) setActiveTab(progress.active_tab);
      if (progress.last_job_id) setActiveJobId(progress.last_job_id);
      progressLoadedRef.current = true;
    }
  }, [progress]);

  // Handle dismissal of Android Web Share Target card
  const handleDismissShared = () => {
    const next = new URLSearchParams(searchParams);
    next.delete('url');
    next.delete('text');
    next.delete('title');
    setSearchParams(next, { replace: true });
    setProductUrlInput('');
  };

  // Track tab changes and persist
  const handleTabChange = (tab: 'twin' | 'capture' | 'measurements') => {
    setActiveTab(tab);
    updateProgress({ active_tab: tab });
  };

  // Track height adjustments and persist
  const handleHeightChange = (val: string) => {
    setHeightCm(val);
    updateProgress({ height_cm: val });
  };

  // Track gender adjustments and persist
  const handleGenderChange = (val: 'neutral' | 'female' | 'male') => {
    setGender(val);
    updateProgress({ gender: val });
  };

  // Track age confirmation and persist
  const handleAdultChange = (confirmed: boolean) => {
    setIsAdult(confirmed);
    if (confirmed) setErrorMessage(null);
    updateProgress({ age_18_confirmed: confirmed });
  };

  // Track photo retention opt-in and persist
  const handleOptInChange = (optIn: boolean) => {
    setOptInKeepFrontPhoto(optIn);
    updateProgress({ opt_in_keep_front_photo: optIn });
  };

  // Query authenticated customer's real twin from backend.
  //
  // `refetch` is exposed (not just `data`) because a failed refetch here is otherwise invisible:
  // React Query keeps showing the *previous* data (often null, from before any capture) when a
  // refetch fails for a reason other than "no twin yet" -- e.g. a transient network error, a 500,
  // a CORS misconfiguration. That looks identical to "no twin yet" in the UI (the Twin Viewer tab
  // would show "No 3D Twin Yet" even though the capture job actually succeeded), which is
  // indistinguishable from a real bug and gives the user no way to retry. See
  // postJobTwinFetchFailed below, which uses this refetch's own result to catch exactly that.
  const { data: twinData, isLoading: isLoadingTwin, refetch: refetchTwin } = useQuery({
    queryKey: ['twin', user?.uid],
    queryFn: async () => {
      try {
        return await apiClient.get<any>('/v1/me/twin');
      } catch (err: any) {
        if (err instanceof ApiError && (err.status === 404 || err.status === 409)) {
          return null;
        }
        throw err;
      }
    },
    enabled: !!signedIn,
  });

  // When backend returns twin data, populate measurements table
  useEffect(() => {
    const zones = twinData?.measurements?.zones;
    if (zones && Object.keys(zones).length > 0) {
      const updated = parseZonesToMeasurements(zones);
      if (updated.length > 0) {
        setMeasurements(updated);
      }
    }
  }, [twinData]);

  // Set once the post-completion twin refetch itself fails, so the Twin Viewer can show a clear
  // "we couldn't load it" state instead of silently falling back to "No 3D Twin Yet" (which
  // otherwise looks identical to "you haven't captured one").
  const [postJobTwinFetchFailed, setPostJobTwinFetchFailed] = useState(false);

  // When live job reaches terminal done, refresh twin data and handle completion/retake/failure
  useEffect(() => {
    if (jobStatus === 'done') {
      queryClient.invalidateQueries({ queryKey: ['twin'] });
      setPostJobTwinFetchFailed(false);
      // invalidateQueries only refetches queries it considers "active"; explicitly refetch this
      // hook's own query too so the viewer is guaranteed to update without needing a page reload.
      refetchTwin()
        .then((result) => {
          if (result.isError) {
            setPostJobTwinFetchFailed(true);
          }
        })
        .catch(() => setPostJobTwinFetchFailed(true));
      setSuccessMessage((prev) => prev || '3D Body Twin created successfully!');
      setErrorMessage(null);
    } else if (jobStatus === 'needs_retake') {
      const msg = job?.error_message || 'Photos require retake. Please follow pose guidelines.';
      setErrorMessage(`Photo Validation Issue: ${msg}. Please capture new photos with clear orthogonal views.`);
      setSuccessMessage(null);
    } else if (jobStatus === 'rejected' || jobStatus === 'failed') {
      const msg = job?.error_message || 'Processing halted. Validation rejected.';
      setErrorMessage(`Job Error: ${msg}. Please check the photo requirements and try again.`);
      setSuccessMessage(null);
    }
  }, [jobStatus, job?.error_message, queryClient, refetchTwin]);

  // Active twin model for display
  const displayTwin = useMemo(() => jobResult || twinData, [jobResult, twinData]);

  // The GLB preview: model-viewer's `src` does its own internal fetch with no Authorization
  // header, so it cannot hit the now-auth-protected GET /api/twin/{id}/glb directly. Fetch it as
  // an authenticated Blob through the existing apiClient instead, and hand model-viewer an
  // in-memory object URL. The effect's cleanup revokes that object URL whenever the twin changes
  // (dependency change runs the previous cleanup before the new fetch) and on unmount, so blob
  // URLs never accumulate.
  const [glbObjectUrl, setGlbObjectUrl] = useState<string | null>(null);
  const [glbLoadError, setGlbLoadError] = useState<string | null>(null);

  useEffect(() => {
    const glbPath: string | undefined = displayTwin?.glb_url;
    let cancelled = false;
    let revoke: (() => void) | null = null;

    setGlbLoadError(null);
    setGlbObjectUrl(null);

    if (!glbPath) {
      return;
    }

    loadObjectUrl(glbPath)
      .then((loaded) => {
        if (cancelled) {
          loaded.revoke();
          return;
        }
        revoke = loaded.revoke;
        setGlbObjectUrl(loaded.url);
      })
      .catch((err: any) => {
        if (cancelled) return;
        const msg =
          err instanceof ApiError
            ? err.status === 401
              ? 'Sign in to view your 3D twin.'
              : err.status === 403
              ? 'This 3D twin does not belong to the signed-in account.'
              : err.message || 'Could not load the 3D model preview.'
            : 'Could not load the 3D model preview.';
        setGlbLoadError(msg);
      });

    return () => {
      cancelled = true;
      if (revoke) {
        revoke();
      }
    };
  }, [displayTwin?.glb_url]);

  // Mutation for creating 3D body twin job via apiClient.
  //
  // POST /api/twin/jobs now returns as soon as the job document exists (status "checking"),
  // not once the full pipeline finishes -- see services/api/main.py::create_twin_job_endpoint.
  // The pipeline runs in the background and can take well over a minute, longer than any
  // reasonable HTTP client timeout (this call used to send `timeoutMs: 120000` to work around
  // that; it no longer needs to, since the response now comes back almost immediately). Once
  // activeJobId is set, useJobStatus(activeJobId) above polls GET /api/twin/jobs/{job_id} and
  // the jobStatus-watching effect above sets the success/error message and refreshes /v1/me/twin
  // once the job actually reaches a terminal state.
  const createTwinMutation = useMutation({
    mutationFn: async (payload: any) => {
      return await apiClient.post<any>('/api/twin/jobs', payload);
    },
    onSuccess: (data) => {
      // Defensive: only ever true if the request fails validation so early that no job is
      // created at all (the current backend doesn't do this, but older/alternate deployments
      // might) -- real validation failures (age, missing photo) are HTTP errors, handled below.
      if (data.status === 'needs_retake' || data.status === 'rejected' || data.status === 'failed') {
        const errorMsg = data.error_message || `Job status: ${data.status} (code: ${data.error_code || 'UNKNOWN'})`;
        setErrorMessage(`Photo Validation Issue: ${errorMsg}`);
        return;
      }

      if (data.job_id) {
        setActiveJobId(data.job_id);
        updateProgress({ last_job_id: data.job_id });
      }
    },
    onError: (err: any) => {
      const msg = err.message || (err.details && (err.details.detail || err.details.error_message)) || 'Could not connect to API service.';
      setErrorMessage(msg);
    },
  });

  const isSubmitting = createTwinMutation.isPending;

  // "Analyze Fit for My Twin" from a shared product URL (Android Share Target card below).
  // Calls the customer-authenticated /v1/me/fit-from-url: it always resolves MY OWN twin
  // server-side from current_uid, never sends any measurement or user id -- the request body is
  // only the product URL. Reuses the backend's existing Reader + Fit Engine pipeline (no new
  // pipeline here), and returns the same verdict-only shape /v1/fit-query already uses.
  const [sharedFitError, setSharedFitError] = useState<string | null>(null);
  const analyzeSharedUrlMutation = useMutation({
    mutationFn: async (productUrl: string) => {
      return await apiClient.post<any>('/v1/me/fit-from-url', { product_url: productUrl });
    },
    onMutate: () => setSharedFitError(null),
    onError: (err: any) => {
      if (err instanceof ApiError && err.status === 409) {
        setSharedFitError('Create your 3D Body Twin first (4-Photo Capture tab), then analyze fit for a shared product.');
        return;
      }
      setSharedFitError(err.message || 'Could not check fit for that product.');
    },
  });

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
      setPhotos((prev) => {
        const next = {
          ...prev,
          [view]: {
            file,
            previewUrl: dataUrl,
            dataUrl,
            fileName: file.name,
          },
        };
        const captured = (Object.keys(VIEW_CONFIG) as ViewType[]).filter((v) => !!next[v].dataUrl);
        updateProgress({ captured_views: captured });
        return next;
      });
    };
    reader.onerror = () => {
      setErrorMessage(`Failed to read image file '${file.name}'.`);
    };
    reader.readAsDataURL(file);
  };

  // Handle removing a photo
  const handleRemovePhoto = (view: ViewType, e: React.MouseEvent) => {
    e.stopPropagation();
    setPhotos((prev) => {
      const next = {
        ...prev,
        [view]: { file: null, previewUrl: null, dataUrl: null, fileName: null },
      };
      const captured = (Object.keys(VIEW_CONFIG) as ViewType[]).filter((v) => !!next[v].dataUrl);
      updateProgress({ captured_views: captured });
      return next;
    });
    if (fileInputRefs[view].current) {
      fileInputRefs[view].current!.value = '';
    }
  };

  // Handle submission to backend workflow
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);
    setJobResult(null);

    // 0. Check authentication
    if (!signedIn) {
      setErrorMessage('Please sign in before submitting photos to generate your private 3D Body Twin.');
      return;
    }

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

    const payload = {
      user_id: user?.uid || ('user_' + Math.random().toString(36).substring(2, 10)),
      height_cm: parsedHeight,
      gender: gender,
      age_18_confirmed: isAdult,
      opt_in_keep_front_photo: optInKeepFrontPhoto,
      photo_front: photos.front.dataUrl,
      photo_back: photos.back.dataUrl,
      photo_left: photos.left.dataUrl,
      photo_right: photos.right.dataUrl,
    };

    createTwinMutation.mutate(payload);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Offline Connectivity Warning Banner */}
      {!isOnline && (
        <div className="p-3.5 rounded-2xl bg-amber-500/10 border border-amber-300 dark:border-amber-700/60 flex items-center space-x-2.5 text-amber-900 dark:text-amber-200 text-xs font-medium animate-fadeIn">
          <WifiOff className="w-4 h-4 flex-shrink-0 text-amber-600 dark:text-amber-400" />
          <span>
            <strong>Offline Mode Active:</strong> You are currently offline. Viewing cached biometric state and measurements. Photo uploads and backend 3D twin calculations will resume once reconnected.
          </span>
        </div>
      )}

      {/* Product Fit Analysis Card (URL input & Android Web Share Target support) */}
      <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-amber-500/10 via-sky-500/10 to-indigo-500/10 border-2 border-amber-300 dark:border-amber-600/60 space-y-3 animate-fadeIn shadow-sm">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500 text-white flex items-center justify-center flex-shrink-0 shadow-sm">
              <Share2 className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
                {sharedUrl ? 'Shared Product Link (Android Share Target)' : 'Product URL Fit Check'}
              </span>
              {sharedTitle && <p className="text-xs font-bold text-slate-900 dark:text-white">{sharedTitle}</p>}
            </div>
          </div>
          {sharedUrl && (
            <button
              type="button"
              onClick={handleDismissShared}
              aria-label="Dismiss shared link"
              className="min-h-[36px] text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 flex items-center space-x-1"
            >
              <X className="w-3.5 h-3.5" />
              <span>Dismiss</span>
            </button>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-2">
          <input
            type="url"
            value={productUrlInput}
            onChange={(e) => setProductUrlInput(e.target.value)}
            placeholder="Paste product URL (e.g. http://localhost:5173/store/shirt.html)"
            className="flex-1 min-h-[44px] px-3.5 py-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-xs font-mono text-slate-900 dark:text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
          />
          {productUrlInput && (
            <a
              href={productUrlInput.startsWith('http') ? productUrlInput : `https://${productUrlInput}`}
              target="_blank"
              rel="noopener noreferrer"
              className="min-h-[44px] px-3.5 py-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-xs font-semibold text-slate-700 dark:text-slate-300 hover:text-sky-600 flex items-center justify-center space-x-1.5 transition-all shadow-sm"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>Open</span>
            </a>
          )}
          <button
            type="button"
            disabled={analyzeSharedUrlMutation.isPending || !productUrlInput.trim()}
            onClick={() => {
              if (!signedIn) {
                setSharedFitError('Sign in first to analyze how this product fits your 3D Body Twin.');
                return;
              }
              analyzeSharedUrlMutation.mutate(productUrlInput.trim());
            }}
            className="min-h-[44px] px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold shadow-md shadow-sky-600/20 flex items-center justify-center space-x-1.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed whitespace-nowrap"
          >
            {analyzeSharedUrlMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            <span>{analyzeSharedUrlMutation.isPending ? 'Checking fit...' : 'Analyze Fit for My Twin'}</span>
          </button>
        </div>

        {/* Actionable Error State with Direct Recovery */}
        {sharedFitError && (
          <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-rose-800 dark:text-rose-200 text-xs animate-fadeIn">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-500" />
              <span>{sharedFitError}</span>
            </div>
            {!signedIn ? (
              <button
                type="button"
                onClick={() => signInWithDemo('demo-shopper')}
                className="min-h-[36px] px-3 py-1 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs whitespace-nowrap shadow-sm"
              >
                Sign In as Demo Shopper
              </button>
            ) : sharedFitError.includes('4-Photo Capture') ? (
              <button
                type="button"
                onClick={() => handleTabChange('capture')}
                className="min-h-[36px] px-3 py-1 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs whitespace-nowrap shadow-sm"
              >
                Go to 4-Photo Capture &rarr;
              </button>
            ) : null}
          </div>
        )}
      </div>

      {analyzeSharedUrlMutation.data && (
        <div className="p-4 sm:p-5 rounded-2xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/60 space-y-3 animate-fadeIn">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">Fit check for the product</h3>
            <button
              type="button"
              onClick={() => analyzeSharedUrlMutation.reset()}
              aria-label="Dismiss fit result"
              className="min-h-[36px] min-w-[36px] p-1.5 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 flex items-center justify-center"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          {analyzeSharedUrlMutation.data.recommended_size ? (
            <>
              <p className="text-sm text-slate-700 dark:text-slate-300">
                Recommended size: <span className="font-bold text-emerald-700 dark:text-emerald-400">{analyzeSharedUrlMutation.data.recommended_size}</span>
                {' '}&bull; {Math.round(analyzeSharedUrlMutation.data.confidence * 100)}% confidence
              </p>
              <div className="flex flex-wrap gap-1.5">
                {analyzeSharedUrlMutation.data.zone_verdicts.map((zv: any) => (
                  <span
                    key={zv.zone}
                    className={`text-[10px] px-2 py-1 rounded-md font-medium capitalize ${
                      zv.verdict === 'ideal'
                        ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                        : zv.verdict === 'tight'
                        ? 'bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-300'
                        : 'bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300'
                    }`}
                  >
                    {zv.zone.replace(/_/g, ' ')}: {zv.verdict}
                  </span>
                ))}
              </div>
              {analyzeSharedUrlMutation.data.alteration_hint && (
                <p className="text-xs text-slate-600 dark:text-slate-400">{analyzeSharedUrlMutation.data.alteration_hint}</p>
              )}
            </>
          ) : (
            <p className="text-sm text-slate-700 dark:text-slate-300">
              None of this product's available sizes fit you well based on your current 3D Body Twin.
            </p>
          )}
        </div>
      )}

      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-sky-500/10 via-indigo-500/10 to-transparent border border-sky-200 dark:border-sky-900/60 shadow-sm">
        <div>
          <h1 className="text-2xl font-black text-slate-900 dark:text-white flex items-center space-x-2">
            <span>Customer 3D Twin &amp; Fit Portal</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-sky-500 text-white font-semibold">PWA</span>
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Deterministic biometric twin derived from 4 orthogonal photos.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* PWA Install Button */}
          {canInstall && (
            <button
              type="button"
              onClick={promptInstall}
              className="min-h-[44px] inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white text-xs font-bold shadow-md shadow-sky-600/25 transition-all active:scale-95"
            >
              <Download className="w-4 h-4" />
              <span>Install App</span>
            </button>
          )}

          {signedIn && user ? (
            <div className="min-h-[44px] inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-sky-50 dark:bg-sky-950/80 border border-sky-200 dark:border-sky-800 text-xs text-sky-800 dark:text-sky-300 font-medium">
              <UserCheck className="w-3.5 h-3.5 text-sky-500" />
              <span>User: {user.isDemo ? `Demo (${user.uid})` : user.uid}</span>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => signInWithDemo('demo-shopper')}
              className="min-h-[44px] inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold shadow transition-all"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign in as Demo Shopper</span>
            </button>
          )}

          {/* Privacy summary badge */}
          <div className="min-h-[44px] inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-300 shadow-sm">
            <ShieldAlert className="w-4 h-4 text-emerald-500" />
            <span>Zero-Retention Raw Photos &bull; Owner-Only Firestore Access</span>
          </div>
        </div>
      </div>

      {/* Live In-Flight Job Status Banner */}
      {activeJobId && !isJobTerminal && (
        <div className="p-4 rounded-2xl bg-sky-50 dark:bg-sky-950/40 border border-sky-300 dark:border-sky-800 space-y-2.5 animate-fadeIn">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-sky-500" />
              <span className="font-bold text-sky-900 dark:text-sky-200">
                3D Fitting Job Active
              </span>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-sky-100 dark:bg-sky-900 text-sky-700 dark:text-sky-300 font-medium">
                {activeJobId}
              </span>
            </div>
            <span className="font-mono text-sky-700 dark:text-sky-300 font-bold">{jobProgressPercent}%</span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
            <div
              className="bg-sky-500 h-full transition-all duration-500 rounded-full"
              style={{ width: `${jobProgressPercent}%` }}
            />
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 font-medium">{jobStage}</p>
        </div>
      )}

      {/* Retake Required / Job Failed Alert Banner */}
      {activeJobId && (jobStatus === 'needs_retake' || jobStatus === 'rejected' || jobStatus === 'failed') && (
        <div className="p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 animate-fadeIn">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-bold text-rose-900 dark:text-rose-200">
                {jobStatus === 'needs_retake' ? 'Photo Retake Required' : '3D Fitting Failed'}
              </p>
              <p className="text-xs text-rose-700 dark:text-rose-300 mt-0.5">
                {job?.error_message || 'The uploaded photos did not pass geometric pose validation.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setActiveJobId(null);
              updateProgress({ last_job_id: null });
              handleTabChange('capture');
            }}
            className="min-h-[44px] px-3.5 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow transition-all whitespace-nowrap"
          >
            Retake Photos
          </button>
        </div>
      )}

      {/* Navigation Tabs with 44px touch targets */}
      <div className="flex space-x-2 border-b border-slate-200 dark:border-slate-800 pb-2">
        <button
          onClick={() => handleTabChange('twin')}
          className={`min-h-[44px] flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === 'twin'
              ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>3D Twin Viewer</span>
        </button>
        <button
          onClick={() => handleTabChange('capture')}
          className={`min-h-[44px] flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === 'capture'
              ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Camera className="w-4 h-4" />
          <span>4-Photo Capture</span>
        </button>
        <button
          onClick={() => handleTabChange('measurements')}
          className={`min-h-[44px] flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
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
            {displayTwin?.glb_url && glbObjectUrl ? (
              <div className="w-full h-80 flex flex-col items-center rounded-xl bg-sky-50/50 dark:bg-slate-900/50 overflow-hidden">
                <model-viewer
                  src={glbObjectUrl}
                  alt={`3D Body Twin ${displayTwin.twin_id || displayTwin.id || 'avatar'}`}
                  camera-controls
                  auto-rotate
                  shadow-intensity="1"
                  style={{ width: '100%', height: '100%' }}
                />
                <div className="w-full flex items-center justify-center gap-2 py-2 bg-slate-900/80 backdrop-blur-md flex-wrap">
                  <span className="text-[11px] px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 font-mono">
                    Twin: {displayTwin.twin_id || displayTwin.id || 'Fitted'}
                  </span>
                  <span className="text-[11px] px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-300 font-mono">
                    Scale: {heightCm} cm
                  </span>
                  {displayTwin.final_iou && (
                    <span className="text-[11px] px-2.5 py-1 rounded-md bg-sky-500/20 text-sky-300 font-mono">
                      IoU: {(displayTwin.final_iou * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
            ) : displayTwin?.glb_url && glbLoadError ? (
              <div className="w-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-rose-300 dark:border-rose-800 rounded-xl bg-rose-50/40 dark:bg-slate-900/50 p-6 text-center">
                <AlertCircle className="w-10 h-10 text-rose-500 mb-3" />
                <h4 className="font-bold text-slate-900 dark:text-white text-lg">3D Model Unavailable</h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mt-1 mb-3">{glbLoadError}</p>
                {!signedIn && (
                  <button
                    type="button"
                    onClick={() => signInWithDemo('demo-shopper')}
                    className="min-h-[44px] px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold shadow transition-all"
                  >
                    Sign in to View Twin
                  </button>
                )}
              </div>
            ) : displayTwin?.glb_url ? (
              <div className="w-full h-80 flex flex-col items-center justify-center rounded-xl bg-sky-50/50 dark:bg-slate-900/50">
                <Loader2 className="w-10 h-10 text-sky-500 animate-spin mb-3" />
                <p className="text-xs text-slate-500 dark:text-slate-400">Loading 3D model preview&hellip;</p>
              </div>
            ) : postJobTwinFetchFailed ? (
              <div className="w-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-rose-300 dark:border-rose-800 rounded-xl bg-rose-50/40 dark:bg-slate-900/50 p-6 text-center">
                <AlertCircle className="w-10 h-10 text-rose-500 mb-3" />
                <h4 className="font-bold text-slate-900 dark:text-white text-lg">Couldn't Load Your 3D Twin</h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mt-1 mb-3">
                  Your capture finished, but we couldn't retrieve the result just now. This is usually a temporary
                  connection issue, not a lost twin.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setPostJobTwinFetchFailed(false);
                    refetchTwin().then((result) => {
                      if (result.isError) setPostJobTwinFetchFailed(true);
                    });
                  }}
                  className="min-h-[44px] px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold shadow transition-all flex items-center space-x-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry</span>
                </button>
              </div>
            ) : (
              <div className="w-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-sky-300 dark:border-sky-800 rounded-xl bg-sky-50/50 dark:bg-slate-900/50 p-6 text-center">
                <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-600 text-white flex items-center justify-center mb-4 shadow-lg shadow-sky-500/30">
                  <Layers className="w-10 h-10 animate-bounce" />
                </div>
                <h4 className="font-bold text-slate-900 dark:text-white text-lg">
                  {isLoadingTwin ? 'Checking for 3D Twin...' : 'No 3D Twin Yet'}
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mt-1">
                  {isLoadingTwin
                    ? 'Loading biometric profile from backend service...'
                    : 'Complete the 4-Photo Capture step to fit your SMPL body twin from orthogonal MediaPipe poses.'}
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
              For the front photo, raise your arms out to about shoulder height so there is a clear
              gap between your upper arms and your body at chest level.
            </p>
          </div>

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-rose-800 dark:text-rose-200 animate-fadeIn">
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
                <div className="text-sm font-medium">{errorMessage}</div>
              </div>
              {!signedIn && (
                <button
                  type="button"
                  onClick={() => signInWithDemo('demo-shopper')}
                  className="min-h-[40px] px-3.5 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold whitespace-nowrap shadow-sm"
                >
                  Sign In as Demo Shopper
                </button>
              )}
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
                onClick={() => handleTabChange('measurements')}
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
                onChange={(e) => handleHeightChange(e.target.value)}
                placeholder="e.g. 175.0"
                className="w-full min-h-[44px] px-3.5 py-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-semibold text-slate-900 dark:text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
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
                    onClick={() => handleGenderChange(g)}
                    className={`min-h-[44px] py-2 px-3 rounded-lg text-xs font-semibold capitalize border transition-all ${
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
            <label className="min-h-[44px] flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={isAdult}
                onChange={(e) => handleAdultChange(e.target.checked)}
                className="h-5 w-5 rounded text-sky-600 focus:ring-sky-500"
              />
              <span className="text-sm font-medium text-amber-900 dark:text-amber-200">
                I confirm that I am 18 years of age or older (Mandatory prior to body capture). <span className="text-rose-500">*</span>
              </span>
            </label>

            <label className="min-h-[44px] flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={optInKeepFrontPhoto}
                onChange={(e) => handleOptInChange(e.target.checked)}
                className="h-5 w-5 rounded text-sky-600 focus:ring-sky-500"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">
                (Optional) Opt-in: Keep only the front photo for virtual try-on previews. By default, raw body photos are automatically deleted upon geometry calculation.
              </span>
            </label>
          </div>

          {/* 4 Photo Upload Slots with Camera Capture */}
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
                    capture="environment"
                    className="hidden"
                  />

                  {photo.previewUrl ? (
                    /* Image Preview Card */
                    <div className="relative rounded-xl border border-sky-400 dark:border-sky-600 bg-slate-900/5 dark:bg-slate-900/60 p-2 flex flex-col items-center justify-between h-52 overflow-hidden group shadow-sm">
                      <div className="w-full flex items-center justify-between z-10 mb-1">
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-emerald-500 text-white text-[10px] font-bold">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>{cfg.label}</span>
                        </span>
                        <div className="flex items-center space-x-1">
                          <button
                            type="button"
                            onClick={() => fileInputRefs[view].current?.click()}
                            className="min-h-[36px] min-w-[36px] p-1.5 rounded-md bg-white/90 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:text-sky-600 shadow text-xs flex items-center justify-center"
                            title="Replace photo"
                          >
                            <RefreshCw className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={(e) => handleRemovePhoto(view, e)}
                            className="min-h-[36px] min-w-[36px] p-1.5 rounded-md bg-white/90 dark:bg-slate-800 text-rose-600 hover:text-rose-700 shadow text-xs flex items-center justify-center"
                            title="Remove photo"
                          >
                            <Trash2 className="w-4 h-4" />
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
                      className="min-h-[190px] p-4 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 flex flex-col items-center justify-center space-y-2 hover:border-sky-500 hover:bg-sky-50/40 dark:hover:bg-slate-900/60 transition-all cursor-pointer bg-slate-50 dark:bg-slate-900/40 text-center"
                    >
                      <div className="w-12 h-12 rounded-full bg-slate-200/70 dark:bg-slate-800 flex items-center justify-center text-slate-500">
                        <Camera className="w-6 h-6" />
                      </div>
                      <div>
                        <span className="text-xs font-bold text-slate-800 dark:text-slate-200 block">{cfg.label}</span>
                        <span className="text-[10px] text-slate-500 dark:text-slate-400 block mt-0.5">{cfg.desc}</span>
                      </div>
                      <span className="text-xs font-semibold text-sky-600 dark:text-sky-400">Tap to capture or upload</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Submit Button with 48px touch target */}
          <button
            type="submit"
            disabled={!isAdult || isSubmitting}
            className={`w-full min-h-[48px] py-3.5 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 ${
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
            ) : !signedIn ? (
              <>
                <LogIn className="w-5 h-5" />
                <span>Sign In to Process 3D Twin</span>
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
