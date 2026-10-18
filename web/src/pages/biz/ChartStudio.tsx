import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { API, DEV_AUTH, useDeskAuth } from '../../lib/useDeskAuth';

// Size chart studio: one flat-lay photo (with an A4 sheet) per size -> measured overlays -> editable table with a grading check
// -> publish, export and put on the store. Everything is stored in cm; the unit toggle only changes what is shown.

const CATEGORIES = ['tshirt', 'shirt', 'kurta', 'dress', 'blouse', 'trousers', 'jeans', 'jacket'] as const;
type Category = (typeof CATEGORIES)[number];
const ZONES: Record<Category, string[]> = {
  tshirt: ['chest', 'shoulder_width', 'garment_length'],
  shirt: ['chest', 'waist', 'shoulder_width', 'sleeve_length', 'garment_length'],
  kurta: ['chest', 'waist', 'hip', 'shoulder_width', 'sleeve_length', 'garment_length'],
  dress: ['chest', 'waist', 'hip', 'shoulder_width', 'garment_length'],
  blouse: ['chest', 'waist', 'shoulder_width', 'sleeve_length', 'garment_length'],
  trousers: ['waist', 'hip', 'thigh', 'inseam'],
  jeans: ['waist', 'hip', 'thigh', 'inseam'],
  jacket: ['chest', 'shoulder_width', 'sleeve_length', 'garment_length'],
};
const TITLES: Record<string, string> = { chest: 'Chest', waist: 'Waist', hip: 'Hip', shoulder_width: 'Shoulder', sleeve_length: 'Sleeve length', garment_length: 'Length', thigh: 'Thigh', inseam: 'Inseam' };
const INCH = 2.54;
const LOW_CONFIDENCE = 0.6;

type Table = Record<string, Record<string, number>>;
interface Flag { size: string; previous_size: string; zone: string; kind: string; severity: string; delta_cm: number; explanation: string; explained_by: string }
interface Note { zone: string; text: string; agreement: string }
interface Slot { file: File | null; preview: string; state: 'empty' | 'ready' | 'measuring' | 'done' | 'failed'; message: string; overlay: string; flags: string[]; confidence: Record<string, number> }
interface Published { id: string; version: number; live: boolean }
interface Exported { json: Record<string, unknown>; html: string; unit: string }

const emptySlot = (): Slot => ({ file: null, preview: '', state: 'empty', message: '', overlay: '', flags: [], confidence: {} });
const label = (z: string) => TITLES[z] ?? z.replace(/_/g, ' ');
const show = (cm: number, unit: 'cm' | 'in') => (unit === 'cm' ? cm : cm / INCH).toFixed(1);
const toBase64 = (file: File) => new Promise<string>((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(String(r.result).split(',')[1]);
  r.onerror = () => reject(new Error('Could not read the file'));
  r.readAsDataURL(file);
});

export const ChartStudio: React.FC = () => {
  const { signedIn, signIn, bearer, error: authError } = useDeskAuth();
  const [devUid, setDevUid] = useState('demo-staff');
  const [category, setCategory] = useState<Category>('shirt');
  const [name, setName] = useState('');
  const [productRef, setProductRef] = useState('');
  const [productUrl, setProductUrl] = useState('');
  const [sizes, setSizes] = useState<string[]>(['S', 'M', 'L', 'XL']);
  const [newSize, setNewSize] = useState('');
  const [slots, setSlots] = useState<Record<string, Slot>>({});
  const [table, setTable] = useState<Table>({});
  const [confidence, setConfidence] = useState<Record<string, Record<string, number>>>({});
  const [edited, setEdited] = useState(false);
  const [unit, setUnit] = useState<'cm' | 'in'>('cm');
  const [flags, setFlags] = useState<Flag[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [published, setPublished] = useState<Published | null>(null);
  const [exported, setExported] = useState<Exported | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [seconds, setSeconds] = useState<number | null>(null);
  const checkTimer = useRef<number | undefined>(undefined);

  const zones = ZONES[category];
  const slot = (s: string): Slot => slots[s] ?? emptySlot();

  const call = useCallback(async (path: string, init?: RequestInit) => {
    const res = await fetch(`${API}${path}`, { ...init, headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${await bearer()}` } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.detail?.message ?? body?.detail?.[0]?.msg ?? `Request failed (${res.status})`);
    return body;
  }, [bearer]);

  const runCheck = useCallback(async (t: Table, explain: boolean) => {
    const filled = Object.fromEntries(Object.entries(t).filter(([, row]) => Object.keys(row).length > 0));
    if (Object.keys(filled).length === 0) { setFlags([]); setNotes([]); return; }
    try {
      const r = await call('/v1/biz/charts/grading-check', { method: 'POST', body: JSON.stringify({ category, sizes: filled, explain, product_url: productUrl.trim() || null }) });
      setFlags(r.flags);
      setNotes(r.review_notes);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [call, category, productUrl]);

  // While the seller edits the table the flags refresh (templates only); the model words them once, right after measuring.
  useEffect(() => {
    if (!edited) return;
    window.clearTimeout(checkTimer.current);
    checkTimer.current = window.setTimeout(() => void runCheck(table, false), 400);
    return () => window.clearTimeout(checkTimer.current);
  }, [table, edited, runCheck]);

  const pick = (s: string, file: File | null) => {
    setSlots((all) => ({ ...all, [s]: file ? { ...emptySlot(), file, preview: URL.createObjectURL(file), state: 'ready' } : emptySlot() }));
    if (file && file.size > 8 * 1024 * 1024) setSlots((all) => ({ ...all, [s]: { ...all[s], state: 'failed', message: 'That file is larger than 8 MB.' } }));
  };

  const measureAll = async () => {
    const todo = sizes.filter((s) => slot(s).file && slot(s).state !== 'failed');
    if (todo.length === 0) { setError('Add at least one photo.'); return; }
    setError(''); setBusy(true); setPublished(null); setExported(null); setSeconds(null); setStartedAt(performance.now());
    const next: Table = { ...table };
    const conf = { ...confidence };
    await Promise.all(todo.map(async (s) => {
      setSlots((all) => ({ ...all, [s]: { ...all[s], state: 'measuring', message: '' } }));
      try {
        const r = await call('/v1/biz/charts/measure', { method: 'POST', body: JSON.stringify({ category, size_label: s, image: await toBase64(slot(s).file!) }) });
        next[s] = Object.fromEntries(Object.entries(r.zones as Record<string, { value_cm: number }>).map(([z, v]) => [z, v.value_cm]));
        conf[s] = Object.fromEntries(Object.entries(r.zones as Record<string, { confidence: number }>).map(([z, v]) => [z, v.confidence]));
        setSlots((all) => ({ ...all, [s]: { ...all[s], state: 'done', overlay: `data:image/png;base64,${r.overlay_png_base64}`, flags: r.flags, message: '' } }));
      } catch (e) {
        setSlots((all) => ({ ...all, [s]: { ...all[s], state: 'failed', message: (e as Error).message } }));
      }
    }));
    setTable(next); setConfidence(conf); setEdited(false);
    await runCheck(next, true);
    setBusy(false);
  };

  const edit = (s: string, z: string, raw: string) => {
    const v = parseFloat(raw);
    setEdited(true);
    setTable((t) => {
      const row = { ...(t[s] ?? {}) };
      if (Number.isNaN(v)) delete row[z]; else row[z] = Math.round((unit === 'cm' ? v : v * INCH) * 10) / 10;
      return { ...t, [s]: row };
    });
  };

  const addSize = () => {
    const s = newSize.trim().toUpperCase();
    if (s && !sizes.includes(s) && sizes.length < 12) setSizes([...sizes, s]);
    setNewSize('');
  };

  const flagged = useMemo(() => new Set(flags.map((f) => `${f.size}|${f.zone}`)), [flags]);

  const publish = async () => {
    setBusy(true); setError('');
    try {
      const filled = Object.fromEntries(Object.entries(table).filter(([, row]) => Object.keys(row).length > 0));
      const p = await call('/v1/biz/charts', { method: 'POST', body: JSON.stringify({
        category, name: name.trim() || `${category} size chart`, product_ref: productRef.trim() || null, sizes: filled, source: edited ? 'edited' : 'measured', confidences: confidence }) });
      setPublished({ id: p.id, version: p.version, live: p.live });
      setExported(await call(`/v1/biz/charts/${p.id}/export?unit=${unit}`));
      if (startedAt !== null) setSeconds(Math.round((performance.now() - startedAt) / 100) / 10);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const reexport = async (u: 'cm' | 'in') => {
    setUnit(u);
    if (published) setExported(await call(`/v1/biz/charts/${published.id}/export?unit=${u}`).catch(() => exported));
  };

  const setLive = async (live: boolean) => {
    if (!published) return;
    try {
      const c = await call(`/v1/biz/charts/${published.id}/${live ? 'use-on-store' : 'remove-from-store'}`, { method: 'POST' });
      setPublished({ ...published, live: c.live });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const copy = (text: string) => void navigator.clipboard?.writeText(text);

  if (!signedIn) {
    return (
      <div className="max-w-md mx-auto px-4 py-16 space-y-4" data-testid="studio-signin">
        <h1 className="text-2xl font-bold">Size chart studio</h1>
        <p className="text-sm text-slate-600 dark:text-slate-300">Sign in with a staff account of your store.</p>
        {DEV_AUTH ? (
          <>
            <label htmlFor="studio-uid" className="block text-sm">Dev sign-in (test mode only)</label>
            <input id="studio-uid" value={devUid} onChange={(e) => setDevUid(e.target.value)} className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2" />
            <button type="button" onClick={() => void signIn(devUid)} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in</button>
          </>
        ) : (
          <button type="button" onClick={() => void signIn()} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in with Google</button>
        )}
        {authError && <p role="alert" className="text-sm text-red-600">{authError}</p>}
      </div>
    );
  }

  const input = 'rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm';
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black">Size chart studio</h1>
          <p className="text-sm text-slate-500">One flat-lay photo per size, with a plain white A4 sheet lying flat close to the garment. Shoot from directly above.</p>
        </div>
        <Link to="/biz" className="text-sm text-sky-600 underline">Business console</Link>
      </div>
      {error && <p role="alert" className="text-sm text-red-600" data-testid="studio-error">{error}</p>}

      <section className="grid md:grid-cols-4 gap-3">
        <label className="text-xs text-slate-500">Category
          <select data-testid="category" value={category} onChange={(e) => { setCategory(e.target.value as Category); setTable({}); setFlags([]); setNotes([]); }} className={`${input} w-full mt-1`}>
            {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="text-xs text-slate-500">Product name
          <input data-testid="product-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Oxford Button-Down Shirt" className={`${input} w-full mt-1`} />
        </label>
        <label className="text-xs text-slate-500">Product reference (for the store)
          <input data-testid="product-ref" value={productRef} onChange={(e) => setProductRef(e.target.value)} placeholder="dtc-shirt" className={`${input} w-full mt-1`} />
        </label>
        <label className="text-xs text-slate-500">Product page URL (optional, for reviews)
          <input data-testid="product-url" value={productUrl} onChange={(e) => setProductUrl(e.target.value)} placeholder="https://..." className={`${input} w-full mt-1`} />
        </label>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-sm font-semibold">Photos</h2>
          <input data-testid="size-new" aria-label="New size" value={newSize} onChange={(e) => setNewSize(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addSize()} placeholder="Add size" className={`${input} w-24`} />
          <button type="button" data-testid="size-add" onClick={addSize} className="text-sm underline text-sky-600">Add</button>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {sizes.map((s) => {
            const sl = slot(s);
            return (
              <div key={s} className="rounded-xl border border-slate-200 dark:border-slate-800 p-3 space-y-2" data-testid={`slot-${s}`}>
                <div className="flex justify-between text-sm font-semibold"><span>Size {s}</span>
                  <button type="button" aria-label={`Remove size ${s}`} onClick={() => { setSizes(sizes.filter((x) => x !== s)); setTable(({ [s]: _gone, ...rest }) => rest); }} className="text-xs text-slate-400">remove</button></div>
                <input data-testid={`photo-${s}`} type="file" accept="image/*" aria-label={`Photo for size ${s}`} onChange={(e) => pick(s, e.target.files?.[0] ?? null)} className="text-xs w-full" />
                {sl.overlay ? <img data-testid={`overlay-${s}`} src={sl.overlay} alt={`Measurement overlay for size ${s}`} className="rounded border border-slate-200 dark:border-slate-800 w-full" />
                  : sl.preview ? <img src={sl.preview} alt={`Photo for size ${s}`} className="rounded border border-slate-200 dark:border-slate-800 w-full opacity-80" /> : null}
                <p data-testid={`status-${s}`} className={`text-xs ${sl.state === 'failed' ? 'text-red-600' : 'text-slate-500'}`}>
                  {sl.state === 'empty' ? 'No photo yet' : sl.state === 'ready' ? 'Ready to measure' : sl.state === 'measuring' ? 'Measuring…' : sl.state === 'done' ? 'Measured' : sl.message}
                  {sl.state === 'done' && sl.flags.length > 0 && <span className="block text-amber-700">Check: {sl.flags.join(', ').replace(/_/g, ' ')}</span>}
                </p>
              </div>
            );
          })}
        </div>
        <button type="button" data-testid="measure-all" disabled={busy} onClick={() => void measureAll()} className="mt-4 rounded-lg bg-sky-600 text-white px-5 py-2 text-sm font-medium disabled:opacity-60">{busy ? 'Working…' : 'Measure all photos'}</button>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Chart ({unit}) <span className="font-normal text-slate-500">edit any value; the grading check updates</span></h2>
          <div role="group" aria-label="Unit" className="text-xs">
            {(['cm', 'in'] as const).map((u) => (
              <button key={u} type="button" data-testid={`unit-${u}`} aria-pressed={unit === u} onClick={() => void reexport(u)}
                className={`px-3 py-1 border border-slate-300 dark:border-slate-700 ${unit === u ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900' : ''} ${u === 'cm' ? 'rounded-l-lg' : 'rounded-r-lg'}`}>{u}</button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="text-sm w-full" data-testid="chart-table">
            <thead><tr className="text-left text-xs text-slate-500"><th className="py-1 pr-3">Size</th>{zones.map((z) => <th key={z} className="px-1">{label(z)} ({unit})</th>)}</tr></thead>
            <tbody>
              {sizes.map((s) => (
                <tr key={s}>
                  <th scope="row" className="pr-3 text-left">{s}</th>
                  {zones.map((z) => {
                    const v = table[s]?.[z];
                    const low = (confidence[s]?.[z] ?? 1) < LOW_CONFIDENCE;
                    return (
                      <td key={z} className="px-1 py-1">
                        <input data-testid={`cell-${s}-${z}`} aria-label={`${s} ${label(z)}`} type="number" step="0.1" value={v === undefined ? '' : show(v, unit)}
                          onChange={(e) => edit(s, z, e.target.value)}
                          title={low ? 'Low confidence: check the overlay' : undefined}
                          className={`w-24 rounded border px-2 py-1 ${flagged.has(`${s}|${z}`) ? 'border-red-500 bg-red-50 dark:bg-red-950/40' : low ? 'border-amber-400' : 'border-slate-300 dark:border-slate-700'} bg-transparent`} />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <div data-testid="flag-list">
          <h2 className="text-sm font-semibold mb-2">Grading check</h2>
          {flags.length === 0 ? <p className="text-sm text-emerald-700 dark:text-emerald-400" data-testid="no-flags">Every zone grows from size to size within the expected range.</p> : (
            <ul className="space-y-2">
              {flags.map((f, i) => (
                <li key={i} data-testid="flag" className={`rounded-lg border p-3 text-sm ${f.severity === 'high' ? 'border-red-300 bg-red-50 dark:bg-red-950/30' : 'border-amber-300 bg-amber-50 dark:bg-amber-950/30'}`}>
                  <span className="text-[11px] uppercase font-semibold mr-2">{f.severity}</span>{f.explanation}
                  <span className="block text-[11px] text-slate-500 mt-1">{f.explained_by === 'model' ? 'Worded by the flash model' : 'Standard wording'}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        {notes.length > 0 && (
          <div data-testid="review-notes">
            <h2 className="text-sm font-semibold mb-2">What reviewers say</h2>
            <ul className="space-y-2">{notes.map((n, i) => <li key={i} data-testid="review-note" className="rounded-lg border border-slate-200 dark:border-slate-800 p-3 text-sm">{n.text}</li>)}</ul>
          </div>
        )}
      </section>

      <section className="space-y-3">
        <div className="flex items-center gap-3">
          <button type="button" data-testid="publish" disabled={busy || Object.keys(table).length === 0} onClick={() => void publish()} className="rounded-lg bg-emerald-600 text-white px-5 py-2 text-sm font-medium disabled:opacity-60">Publish chart</button>
          {seconds !== null && <span data-testid="elapsed" className="text-xs text-slate-500">From the first photo to a published chart: {seconds} s</span>}
        </div>
        {published && exported && (
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 p-4 space-y-3" data-testid="published">
            <p className="text-sm">Published as version {published.version} <span className="text-xs text-slate-500" data-testid="published-id">{published.id}</span>.</p>
            <div className="flex flex-wrap gap-2">
              {published.live
                ? <button type="button" data-testid="remove-from-store" onClick={() => void setLive(false)} className="rounded-lg border border-slate-300 dark:border-slate-700 px-4 py-2 text-sm">Remove from store</button>
                : <button type="button" data-testid="use-on-store" onClick={() => void setLive(true)} className="rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 px-4 py-2 text-sm font-medium">Use on store</button>}
              {published.live && <span data-testid="live-note" className="text-sm text-emerald-700 dark:text-emerald-400 self-center">Live on the store{productRef ? ` for ${productRef}` : ''}.</span>}
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between text-xs mb-1"><span className="font-semibold">JSON</span><button type="button" onClick={() => copy(JSON.stringify(exported.json, null, 2))} className="underline text-sky-600">Copy</button></div>
                <textarea data-testid="export-json" readOnly rows={10} value={JSON.stringify(exported.json, null, 2)} className={`${input} w-full font-mono text-xs`} />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1"><span className="font-semibold">HTML table</span><button type="button" onClick={() => copy(exported.html)} className="underline text-sky-600">Copy</button></div>
                <textarea data-testid="export-html" readOnly rows={10} value={exported.html} className={`${input} w-full font-mono text-xs`} />
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default ChartStudio;
