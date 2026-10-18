import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { API, DEV_AUTH, useDeskAuth } from '../../lib/useDeskAuth';

// Returns desk: the routing rules settle what they can, a person decides the rest. There is deliberately no "reject".

interface Card {
  id: string;
  created_at: string;
  status: string;
  reason: string;
  note: string | null;
  order_id: string;
  sku: string;
  product_title: string;
  ordered_size: string;
  guarantee: boolean;
  twin_verdict: { recommended_size: string | null; confidence: number | null };
  routing: { route: string; exchange_size: string | null; seller_label_error: boolean; model_miss: boolean };
  routing_reason: string;
  decision_log: { rule: string; text: string; actor?: string }[];
  evidence: {
    overlay_image_url: string | null;
    source: string | null;
    spec_cm: Record<string, number>;
    measured_cm: Record<string, number>;
    diff_cm: Record<string, number>;
  };
  summary: { text: string; source: string } | null;
  simulated: boolean;
}

interface Kpis {
  orders: number;
  guaranteed_orders: number;
  guaranteed_share: number;
  returns: number;
  open_returns: number;
  exchanges: number;
  refunds: number;
  nudges_shown: number;
  nudges_accepted: number;
  nudges_accept_rate: number;
  seller_label_errors: number;
  seller_label_error_skus: string[];
}

const label = (s: string) => s.replace(/_/g, ' ');
const pct = (n: number) => `${Math.round(n * 100)}%`;
const STATUS_STYLE: Record<string, string> = {
  pending_review: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
  in_review: 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300',
  approved_exchange: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
  approved_refund: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
};

export const ReturnsDesk: React.FC = () => {
  const { signedIn, signIn, bearer, error: authError } = useDeskAuth();
  const [devUid, setDevUid] = useState('demo-staff');
  const [cards, setCards] = useState<Card[]>([]);
  const [kpis, setKpis] = useState<Kpis | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const call = useCallback(
    async (path: string, init?: RequestInit) => {
      const res = await fetch(`${API}${path}`, {
        ...init,
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${await bearer()}`, ...(init?.headers ?? {}) },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail?.message ?? `Request failed (${res.status})`);
      }
      return res.json();
    },
    [bearer],
  );

  const refresh = useCallback(async () => {
    try {
      const [list, k] = await Promise.all([call('/v1/biz/returns'), call('/v1/biz/kpis')]);
      setCards(list);
      setKpis(k);
      setSelected((cur) => cur ?? list[0]?.id ?? null);
      setError('');
    } catch (e) {
      setError((e as Error).message);
    }
  }, [call]);

  useEffect(() => {
    if (signedIn) void refresh();
  }, [signedIn, refresh]);

  const current = cards.find((c) => c.id === selected) ?? null;
  const replace = (c: Card) => setCards((all) => all.map((x) => (x.id === c.id ? c : x)));

  const act = async (action: 'approve' | 'exchange' | 'review') => {
    if (!current) return;
    setBusy(true);
    try {
      replace(await call(`/v1/biz/returns/${current.id}/decision`, { method: 'POST', body: JSON.stringify({ action, note: note.trim() || null }) }));
      setNote('');
      setKpis(await call('/v1/biz/kpis'));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const summarize = async (refreshIt = false) => {
    if (!current) return;
    setBusy(true);
    try {
      replace(await call(`/v1/biz/returns/${current.id}/summary${refreshIt ? '?refresh=true' : ''}`, { method: 'POST' }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!signedIn) {
    return (
      <div className="max-w-md mx-auto px-4 py-16 space-y-4" data-testid="desk-signin">
        <h1 className="text-2xl font-bold">Returns desk</h1>
        <p className="text-sm text-slate-600 dark:text-slate-300">Sign in with a staff account of your store.</p>
        {DEV_AUTH ? (
          <>
            <label htmlFor="desk-uid" className="block text-sm">Dev sign-in (test mode only)</label>
            <input id="desk-uid" value={devUid} onChange={(e) => setDevUid(e.target.value)} className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2" />
            <button type="button" onClick={() => void signIn(devUid)} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in</button>
          </>
        ) : (
          <button type="button" onClick={() => void signIn()} className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in with Google</button>
        )}
        {authError && <p role="alert" className="text-sm text-red-600">{authError}</p>}
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black">Returns desk</h1>
          <p className="text-sm text-slate-500">The rules settle what they can. A person decides the rest. Nothing is rejected automatically.</p>
        </div>
        <Link to="/biz" className="text-sm text-sky-600 underline">Business console</Link>
      </div>

      {error && <p role="alert" className="text-sm text-red-600" data-testid="desk-error">{error}</p>}

      {kpis && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4" data-testid="kpi-strip">
          <Kpi title="Guaranteed orders" value={pct(kpis.guaranteed_share)} sub={`${kpis.guaranteed_orders} of ${kpis.orders} orders`} id="kpi-guaranteed" />
          <Kpi title="Exchanges vs refunds" value={`${kpis.exchanges} : ${kpis.refunds}`} sub={`${kpis.open_returns} still open`} id="kpi-exchanges" />
          <Kpi title="Nudges accepted" value={`${kpis.nudges_accepted} of ${kpis.nudges_shown}`} sub={kpis.nudges_shown ? pct(kpis.nudges_accept_rate) : 'none shown yet'} id="kpi-nudges" />
          <Kpi title="Seller label errors" value={String(kpis.seller_label_errors)} sub={kpis.seller_label_error_skus.join(', ') || 'none'} id="kpi-labels" />
        </div>
      )}

      <div className="grid lg:grid-cols-[320px_1fr] gap-6">
        <ul className="space-y-2" data-testid="return-list">
          {cards.length === 0 && <li className="text-sm text-slate-500">No returns yet.</li>}
          {cards.map((c) => (
            <li key={c.id}>
              <button type="button" onClick={() => setSelected(c.id)} data-testid="return-row"
                className={`w-full text-left rounded-xl border p-3 ${c.id === selected ? 'border-sky-500 bg-sky-50 dark:bg-sky-950/40' : 'border-slate-200 dark:border-slate-800'}`}>
                <div className="font-medium text-sm">{c.product_title} · {c.ordered_size}</div>
                <div className="text-xs text-slate-500">{label(c.reason)} · {label(c.routing.route)}</div>
                <span className={`inline-block mt-1 text-[11px] px-2 py-0.5 rounded-full ${STATUS_STYLE[c.status] ?? ''}`}>{label(c.status)}</span>
              </button>
            </li>
          ))}
        </ul>

        {current && (
          <article className="rounded-2xl border border-slate-200 dark:border-slate-800 p-5 space-y-5" data-testid="evidence-card">
            <header className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-bold">{current.product_title}, size {current.ordered_size}</h2>
                <p className="text-xs text-slate-500">Order {current.order_id.slice(0, 8)} · reported: {label(current.reason)} · {current.guarantee ? 'fit guarantee' : 'no fit guarantee'}</p>
              </div>
              <span data-testid="card-status" className={`text-xs px-2.5 py-1 rounded-full ${STATUS_STYLE[current.status] ?? ''}`}>{label(current.status)}</span>
            </header>
            {current.simulated && <p className="text-xs text-amber-700 dark:text-amber-300">Simulated unboxing data for the demo.</p>}

            <section className="grid md:grid-cols-2 gap-5">
              <div>
                <h3 className="text-sm font-semibold mb-2">Unboxing overlay</h3>
                {current.evidence.overlay_image_url ? (
                  <img src={current.evidence.overlay_image_url} alt="Unboxing measurement overlay" className="rounded-lg border border-slate-200 dark:border-slate-800 w-full bg-slate-100" />
                ) : (
                  <p className="text-sm text-slate-500">No unboxing measurement was provided.</p>
                )}
              </div>
              <div>
                <h3 className="text-sm font-semibold mb-2">Ordered size spec vs measured (cm)</h3>
                <table className="w-full text-sm" data-testid="spec-table">
                  <thead><tr className="text-left text-xs text-slate-500"><th>Zone</th><th>Spec</th><th>Measured</th><th>Diff</th></tr></thead>
                  <tbody>
                    {Object.keys({ ...current.evidence.spec_cm, ...current.evidence.measured_cm }).map((z) => (
                      <tr key={z}>
                        <td className="capitalize">{label(z)}</td>
                        <td>{current.evidence.spec_cm[z]?.toFixed(1) ?? 'n/a'}</td>
                        <td>{current.evidence.measured_cm[z]?.toFixed(1) ?? 'n/a'}</td>
                        <td className={Math.abs(current.evidence.diff_cm[z] ?? 0) > 1.5 ? 'text-red-600 font-medium' : ''}>
                          {current.evidence.diff_cm[z] === undefined ? '' : `${current.evidence.diff_cm[z] > 0 ? '+' : ''}${current.evidence.diff_cm[z].toFixed(1)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-xs text-slate-500 mt-3" data-testid="twin-verdict">
                  Twin verdict at purchase: {current.twin_verdict.recommended_size ? `size ${current.twin_verdict.recommended_size}` : 'no size'}
                  {current.twin_verdict.confidence !== null && `, confidence ${current.twin_verdict.confidence.toFixed(2)}`}
                </p>
              </div>
            </section>

            <section>
              <h3 className="text-sm font-semibold mb-1">Routing</h3>
              <p className="text-sm" data-testid="routing-reason"><strong className="capitalize">{label(current.routing.route)}.</strong> {current.routing_reason}</p>
              {current.routing.seller_label_error && <p className="text-xs mt-1 text-amber-700 dark:text-amber-300">SKU {current.sku} is flagged for a label error.</p>}
              {current.routing.model_miss && <p className="text-xs mt-1 text-slate-500">Logged as a size recommendation miss for evaluation.</p>}
              <details className="mt-2"><summary className="text-xs cursor-pointer text-slate-500">Decision log</summary>
                <ol className="text-xs mt-2 space-y-1 list-decimal pl-5" data-testid="decision-log">{current.decision_log.map((e, i) => <li key={i}>{e.text}</li>)}</ol>
              </details>
              {current.note && <p className="text-xs mt-2 text-slate-500">Shopper note: {current.note}</p>}
            </section>

            <section>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-sm font-semibold">Case summary</h3>
                <button type="button" disabled={busy} onClick={() => void summarize(!!current.summary)} className="text-xs underline text-sky-600" data-testid="summarize">{current.summary ? 'Regenerate' : 'Generate summary'}</button>
              </div>
              {current.summary ? (
                <p className="text-sm" data-testid="summary-text">{current.summary.text}<span className="block text-[11px] text-slate-400 mt-1">{current.summary.source === 'model' ? 'Written by the flash model from the facts above.' : 'Template summary from the facts above.'}</span></p>
              ) : <p className="text-sm text-slate-500">Facts and numbers only. No judgement of the shopper.</p>}
            </section>

            <section className="space-y-2">
              <label htmlFor="decision-note" className="text-xs text-slate-500">Note for the log (optional)</label>
              <input id="decision-note" value={note} onChange={(e) => setNote(e.target.value)} className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm" />
              <div className="flex flex-wrap gap-2">
                <button type="button" disabled={busy} onClick={() => void act('approve')} className="rounded-lg bg-emerald-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-60">Approve return</button>
                <button type="button" disabled={busy} onClick={() => void act('exchange')} className="rounded-lg bg-sky-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-60">Exchange</button>
                <button type="button" disabled={busy} onClick={() => void act('review')} className="rounded-lg border border-slate-300 dark:border-slate-700 px-4 py-2 text-sm font-medium disabled:opacity-60">Send to review</button>
              </div>
            </section>
          </article>
        )}
      </div>
    </div>
  );
};

const Kpi: React.FC<{ title: string; value: string; sub: string; id: string }> = ({ title, value, sub, id }) => (
  <div className="rounded-2xl border border-slate-200 dark:border-slate-800 p-4" data-testid={id}>
    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</p>
    <p className="text-2xl font-black mt-1">{value}</p>
    <p className="text-xs text-slate-500 mt-1">{sub}</p>
  </div>
);

export default ReturnsDesk;
