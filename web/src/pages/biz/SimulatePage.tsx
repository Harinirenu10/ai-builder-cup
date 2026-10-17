import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { API, DEV_AUTH, useDeskAuth } from '../../lib/useDeskAuth';

// Pre-production return simulation: a size chart against a synthetic population. The banner is permanent on purpose:
// the population is not real people until it has been calibrated with consented, anonymized twins.

const DEFAULT_BANNER = 'Synthetic population from a statistical stand-in with invented body-measurement distributions, not the SMPL shape space. Calibrate with consented, anonymized twins before relying on it.';
const CATEGORIES = ['tshirt', 'shirt', 'kurta', 'dress', 'blouse', 'trousers', 'jeans', 'jacket'];
const FITS = ['slim', 'regular', 'relaxed', 'oversized'];
const FABRICS: Record<string, { fiber: string; percent: number }[]> = {
  'cotton': [{ fiber: 'cotton', percent: 100 }],
  'rayon (viscose)': [{ fiber: 'rayon', percent: 100 }],
  'polyester': [{ fiber: 'polyester', percent: 100 }],
  'cotton and 2% elastane': [{ fiber: 'cotton', percent: 98 }, { fiber: 'elastane', percent: 2 }],
};
const EXAMPLES: Record<string, { label: string; category: string; chart: Chart }> = {
  shirt: { label: 'Example: shirt (garment chart, cm)', category: 'shirt', chart: { chart_type: 'garment', unit: 'cm', sizes: {
    S: { chest: 108, waist: 104, shoulder_width: 44, sleeve_length: 60, garment_length: 72 }, M: { chest: 114, waist: 110, shoulder_width: 46, sleeve_length: 61.5, garment_length: 74 },
    L: { chest: 120, waist: 116, shoulder_width: 48, sleeve_length: 63, garment_length: 76 }, XL: { chest: 126, waist: 122, shoulder_width: 50, sleeve_length: 64.5, garment_length: 78 } } } },
  kurta: { label: 'Example: kurta (body chart, inches)', category: 'kurta', chart: { chart_type: 'body', unit: 'in', sizes: {
    S: { chest: 34.5, waist: 28.5, hip: 36.5 }, M: { chest: 36.5, waist: 30.5, hip: 38.5 }, L: { chest: 38.5, waist: 32.5, hip: 40.5 },
    XL: { chest: 40.5, waist: 34.5, hip: 42.5 }, XXL: { chest: 42.5, waist: 36.5, hip: 44.5 } } } },
};

interface Chart { chart_type: 'body' | 'garment'; unit: 'cm' | 'in'; sizes: Record<string, Record<string, number>> }
interface Share { low: number; ideal: number; high: number; low_word: string; high_word: string }
interface Change { size: string; zone: string; from_cm: number; to_cm: number; delta_cm: number; coverage_gain_alone: number }
interface Result {
  n_bodies: number; population_source: string; reweighted: boolean; banner: string; category: string; fit_type: string; chart_type: string;
  sizes: string[]; size_mix: Record<string, number>; no_recommendation_share: number; no_good_size_share: number; coverage: number;
  zone_fit: Record<string, Record<string, Share>>; seconds: number;
  regrade: { changes: Change[]; coverage_before: number; coverage_after: number; grading_flags_after: unknown[] } | null;
}
interface Memo { headline: string; sections: { heading: string; body: string }[]; source: string }
interface Published { id: string; name: string; category: string; version: number }

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const label = (z: string) => z.replace(/_/g, ' ');

export const SimulatePage: React.FC = () => {
  const { signedIn, signIn, bearer, error: authError } = useDeskAuth();
  const [devUid, setDevUid] = useState('demo-staff');
  const [published, setPublished] = useState<Published[]>([]);
  const [chartId, setChartId] = useState('');
  const [inline, setInline] = useState<Chart | null>(null);
  const [source, setSource] = useState('');
  const [category, setCategory] = useState('shirt');
  const [fit, setFit] = useState('regular');
  const [fabric, setFabric] = useState('cotton');
  const [stretch, setStretch] = useState('none');
  const [preShrunk, setPreShrunk] = useState('unknown');
  const [target, setTarget] = useState(false);
  const [regrade, setRegrade] = useState(true);
  const [result, setResult] = useState<Result | null>(null);
  const [memo, setMemo] = useState<Memo | null>(null);
  const [memoState, setMemoState] = useState<'idle' | 'writing' | 'done'>('idle');
  const [facts, setFacts] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const call = useCallback(async (path: string, body?: unknown) => {
    const res = await fetch(`${API}${path}`, { method: body === undefined ? 'GET' : 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${await bearer()}` }, body: body === undefined ? undefined : JSON.stringify(body) });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(json?.detail?.message ?? json?.detail?.[0]?.msg ?? `Request failed (${res.status})`);
    return json;
  }, [bearer]);

  useEffect(() => {
    if (signedIn) call('/v1/biz/charts').then(setPublished).catch(() => setPublished([]));
  }, [signedIn, call]);

  const choosePublished = (id: string) => {
    setChartId(id); setInline(null);
    const p = published.find((c) => c.id === id);
    if (p) { setCategory(p.category); setSource(`${p.name} (version ${p.version})`); }
  };

  const useExample = (key: string) => {
    const e = EXAMPLES[key];
    setInline(e.chart); setChartId(''); setCategory(e.category); setSource(e.label.replace('Example: ', 'example '));
  };

  const readFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      const j = JSON.parse(await file.text());
      const sizes = j.sizes ?? j.json?.sizes;
      if (!sizes || typeof sizes !== 'object') throw new Error('The file has no "sizes".');
      setInline({ sizes, chart_type: j.chart_type === 'body' ? 'body' : 'garment', unit: j.unit === 'in' ? 'in' : 'cm' });
      setChartId(''); setSource(file.name);
      if (CATEGORIES.includes(j.category)) setCategory(j.category);
      setError('');
    } catch (e) {
      setError(`Could not read that chart: ${(e as Error).message}`);
    }
  };

  const request = () => ({
    ...(chartId ? { chart_id: chartId } : { chart: inline }), category, fit_type: fit, fabric: FABRICS[fabric], stretch_class: stretch,
    pre_shrunk: preShrunk === 'unknown' ? null : preShrunk === 'yes', target, regrade,
  });

  const run = async () => {
    if (!chartId && !inline) { setError('Pick a published chart, upload one, or load an example.'); return; }
    setBusy(true); setError(''); setResult(null); setMemo(null); setMemoState('idle'); setFacts('');
    try {
      const body = request();
      setResult(await call('/v1/biz/simulate', body));
      setBusy(false);
      setMemoState('writing');                       // the statistics are on screen; the memo follows when the model is done
      call('/v1/biz/simulate/memo', body).then((r) => { setMemo(r.memo); setMemoState('done'); }).catch((e) => { setError((e as Error).message); setMemoState('idle'); });
    } catch (e) {
      setError((e as Error).message); setBusy(false);
    }
  };

  const loadFacts = async () => setFacts(JSON.stringify(await call('/v1/biz/simulate/facts', request()), null, 2));

  const banner = (
    <div role="note" data-testid="sim-banner" className="sticky top-0 z-30 border-b border-amber-300 bg-amber-100 dark:bg-amber-950 dark:border-amber-800 px-4 py-2 text-sm text-amber-900 dark:text-amber-200">
      <strong>Synthetic data. </strong>{result?.banner ?? DEFAULT_BANNER}
    </div>
  );

  if (!signedIn) {
    return (
      <div>
        {banner}
        <div className="max-w-md mx-auto px-4 py-16 space-y-4" data-testid="sim-signin">
          <h1 className="text-2xl font-bold">Return simulation</h1>
          {DEV_AUTH ? (
            <>
              <label htmlFor="sim-uid" className="block text-sm">Dev sign-in (test mode only)</label>
              <input id="sim-uid" value={devUid} onChange={(e) => setDevUid(e.target.value)} className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2" />
              <button type="button" onClick={() => void signIn(devUid)} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in</button>
            </>
          ) : <button type="button" onClick={() => void signIn()} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in with Google</button>}
          {authError && <p role="alert" className="text-sm text-red-600">{authError}</p>}
        </div>
      </div>
    );
  }

  const field = 'rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm w-full mt-1';
  const mixRows = result ? [...result.sizes.map((s) => [s, result.size_mix[s]] as const), ['No size recommended', result.no_recommendation_share] as const] : [];
  return (
    <div>
      {banner}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-black">Return simulation</h1>
            <p className="text-sm text-slate-500">Run a size chart against a synthetic population before you produce it.</p>
          </div>
          <Link to="/biz" className="text-sm text-sky-600 underline">Business console</Link>
        </div>
        {error && <p role="alert" className="text-sm text-red-600" data-testid="sim-error">{error}</p>}

        <section className="grid md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <h2 className="text-sm font-semibold">Chart</h2>
            <label className="text-xs text-slate-500 block">Published chart
              <select data-testid="chart-pick" value={chartId} onChange={(e) => choosePublished(e.target.value)} className={field}>
                <option value="">None</option>
                {published.map((c) => <option key={c.id} value={c.id}>{c.name} (version {c.version})</option>)}
              </select>
            </label>
            <label className="text-xs text-slate-500 block">Or upload a chart (JSON with sizes)
              <input data-testid="chart-file" type="file" accept="application/json,.json" onChange={(e) => void readFile(e.target.files?.[0])} className="text-xs mt-1 w-full" />
            </label>
            <div className="flex flex-wrap gap-2">
              {Object.entries(EXAMPLES).map(([k, e]) => <button key={k} type="button" data-testid={`example-${k}`} onClick={() => useExample(k)} className="text-xs underline text-sky-600">{e.label}</button>)}
            </div>
            <p className="text-xs text-slate-500" data-testid="chart-source">{source ? `Using ${source}.` : 'No chart chosen yet.'}</p>
          </div>
          <div className="space-y-2">
            <h2 className="text-sm font-semibold">Garment</h2>
            <label className="text-xs text-slate-500 block">Category
              <select data-testid="sim-category" value={category} onChange={(e) => setCategory(e.target.value)} className={field}>{CATEGORIES.map((c) => <option key={c}>{c}</option>)}</select></label>
            <label className="text-xs text-slate-500 block">Fit
              <select data-testid="sim-fit" value={fit} onChange={(e) => setFit(e.target.value)} className={field}>{FITS.map((c) => <option key={c}>{c}</option>)}</select></label>
            <label className="text-xs text-slate-500 block">Fabric
              <select data-testid="sim-fabric" value={fabric} onChange={(e) => setFabric(e.target.value)} className={field}>{Object.keys(FABRICS).map((c) => <option key={c}>{c}</option>)}</select></label>
          </div>
          <div className="space-y-2">
            <h2 className="text-sm font-semibold">Options</h2>
            <label className="text-xs text-slate-500 block">Stretch
              <select value={stretch} onChange={(e) => setStretch(e.target.value)} className={field}>{['none', 'low', 'high'].map((c) => <option key={c}>{c}</option>)}</select></label>
            <label className="text-xs text-slate-500 block">Pre-shrunk
              <select value={preShrunk} onChange={(e) => setPreShrunk(e.target.value)} className={field}>{['unknown', 'yes', 'no'].map((c) => <option key={c}>{c}</option>)}</select></label>
            <label className="text-sm flex gap-2 items-center"><input type="checkbox" checked={target} onChange={(e) => setTarget(e.target.checked)} /> Reweight to the target market</label>
            <label className="text-sm flex gap-2 items-center"><input type="checkbox" checked={regrade} onChange={(e) => setRegrade(e.target.checked)} /> Suggest regrades</label>
            <button type="button" data-testid="sim-run" disabled={busy} onClick={() => void run()} className="rounded-lg bg-sky-600 text-white px-5 py-2 text-sm font-medium disabled:opacity-60 w-full">{busy ? 'Simulating…' : 'Run simulation'}</button>
          </div>
        </section>

        {result && (
          <>
            <section className="grid grid-cols-2 lg:grid-cols-4 gap-4" data-testid="sim-kpis">
              <Kpi id="kpi-covered" title="Bodies with a good size" value={pct(result.coverage)} sub="every hard-to-alter zone ideal, as bought and after a wash" />
              <Kpi id="kpi-nogood" title="No good size" value={pct(result.no_good_size_share)} sub="in any size of this chart" />
              <Kpi id="kpi-none" title="No size recommended" value={pct(result.no_recommendation_share)} sub="a hard zone fails in every size" />
              <Kpi id="kpi-run" title="Population" value={`${result.n_bodies}`} sub={`${result.population_source}${result.reweighted ? ', reweighted' : ''}, run in ${result.seconds.toFixed(1)} s`} />
            </section>

            <section data-testid="size-mix">
              <h2 className="text-sm font-semibold mb-2">Best size for each body (size mix)</h2>
              <ul className="space-y-1">
                {mixRows.map(([name, v]) => (
                  <li key={name} className="flex items-center gap-3 text-sm" data-testid={`mix-${name}`}>
                    <span className="w-40 shrink-0">{name}</span>
                    <span className="flex-1 h-4 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden"><span className={`block h-full ${name === 'No size recommended' ? 'bg-slate-400' : 'bg-sky-500'}`} style={{ width: `${Math.min(100, v * 100)}%` }} /></span>
                    <span className="w-16 text-right tabular-nums">{pct(v)}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section data-testid="zone-fit">
              <h2 className="text-sm font-semibold mb-1">Where each size misses (among the bodies it is the best size for)</h2>
              <p className="text-xs text-slate-500 mb-2">Below the zone's range, inside it, above it: as bought or after the first wash.</p>
              <div className="overflow-x-auto"><table className="text-sm w-full">
                <thead><tr className="text-left text-xs text-slate-500"><th>Size</th><th>Zone</th><th className="w-1/2">Below / ideal / above</th><th>Below</th><th>Above</th></tr></thead>
                <tbody>
                  {result.sizes.flatMap((s) => Object.entries(result.zone_fit[s] ?? {}).map(([z, sh]) => (
                    <tr key={`${s}-${z}`} data-testid={`zone-${s}-${z}`}>
                      <td>{s}</td><td className="capitalize">{label(z)}</td>
                      <td><span className="flex h-3 rounded overflow-hidden bg-slate-100 dark:bg-slate-800" role="img" aria-label={`${pct(sh.low)} ${sh.low_word}, ${pct(sh.ideal)} ideal, ${pct(sh.high)} ${sh.high_word}`}>
                        <span className="bg-amber-400" style={{ width: `${sh.low * 100}%` }} /><span className="bg-emerald-500" style={{ width: `${sh.ideal * 100}%` }} /><span className="bg-sky-400" style={{ width: `${sh.high * 100}%` }} /></span></td>
                      <td className="tabular-nums">{pct(sh.low)} {sh.low_word}</td><td className="tabular-nums">{pct(sh.high)} {sh.high_word}</td>
                    </tr>
                  )))}
                </tbody>
              </table></div>
            </section>

            {result.regrade && (
              <section data-testid="regrade">
                <h2 className="text-sm font-semibold mb-1">Regrade suggestions (up to 2 cm per zone per size)</h2>
                <p className="text-sm" data-testid="regrade-summary">Bodies with a good size: {pct(result.regrade.coverage_before)} now, {pct(result.regrade.coverage_after)} with these changes.</p>
                {result.regrade.changes.length === 0 ? <p className="text-sm text-slate-500">No change of up to 2 cm raises coverage.</p> : (
                  <table className="text-sm mt-2" data-testid="regrade-table"><thead><tr className="text-left text-xs text-slate-500"><th className="pr-4">Size</th><th className="pr-4">Zone</th><th className="pr-4">Now</th><th className="pr-4">Suggested</th><th>Coverage gained alone</th></tr></thead>
                    <tbody>{result.regrade.changes.map((c, i) => <tr key={i}><td>{c.size}</td><td className="capitalize">{label(c.zone)}</td><td>{c.from_cm.toFixed(1)} cm</td><td>{c.to_cm.toFixed(1)} cm ({c.delta_cm > 0 ? '+' : ''}{c.delta_cm.toFixed(1)})</td><td>{c.coverage_gain_alone >= 0 ? '+' : ''}{pct(c.coverage_gain_alone)}</td></tr>)}</tbody></table>
                )}
              </section>
            )}

            <section data-testid="memo-section" className="rounded-2xl border border-slate-200 dark:border-slate-800 p-5 space-y-3">
              <h2 className="text-sm font-semibold">Merchandiser memo</h2>
              {memoState === 'writing' && <p className="text-sm text-slate-500" data-testid="memo-writing">Writing the memo from these numbers. This can take up to a minute; the statistics above are final.</p>}
              {memo && (
                <article data-testid="memo" className="space-y-3">
                  <h3 className="text-lg font-bold" data-testid="memo-headline">{memo.headline}</h3>
                  {memo.sections.map((s, i) => <div key={i}><h4 className="text-sm font-semibold">{s.heading}</h4><p className="text-sm whitespace-pre-line">{s.body}</p></div>)}
                  <p className="text-[11px] text-slate-400" data-testid="memo-source">{memo.source === 'model' ? 'Written by the pro model from the statistics above; every number was checked against them.' : 'Template written from the statistics above (the model was not used or its text failed the checks).'}</p>
                </article>
              )}
              <button type="button" onClick={() => void loadFacts()} className="text-xs underline text-sky-600" data-testid="show-facts">Show the statistics the memo is written from</button>
              {facts && <pre className="text-xs bg-slate-50 dark:bg-slate-900 rounded p-3 overflow-x-auto max-h-64" data-testid="facts">{facts}</pre>}
            </section>
          </>
        )}
      </div>
    </div>
  );
};

const Kpi: React.FC<{ id: string; title: string; value: string; sub: string }> = ({ id, title, value, sub }) => (
  <div className="rounded-2xl border border-slate-200 dark:border-slate-800 p-4" data-testid={id}>
    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</p>
    <p className="text-2xl font-black mt-1">{value}</p>
    <p className="text-xs text-slate-500 mt-1">{sub}</p>
  </div>
);

export default SimulatePage;
