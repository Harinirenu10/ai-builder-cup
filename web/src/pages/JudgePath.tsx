import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2, ChevronRight, Loader2, ShieldCheck } from 'lucide-react';
import '@google/model-viewer';
import { api, currentSession } from '../lib/demoAuth';

// The judge path: five steps on one page, all on the signed-in sandbox persona. Everything shown is SIMULATED and says so.

interface Twin { simulated: boolean; note: string; body_model: string; height_cm: number; zones: Record<string, { value_cm: number }>; glb_url: string; privacy: string }
interface ProductCard { ref: string; slug: string; title: string; price: string; recommended_size: string | null; confidence: number }
interface ZoneRow { zone: string; hard: boolean; verdict: string; verdict_after_wash: string; ease_now_cm: number; pose_strain: string[] }
interface Fit {
  title: string; recommended_size: string | null; shown_size: string; confidence: number; zones: ZoneRow[]; warnings: string[]; alterations: string[];
  facts: string[]; review_notes: string[]; heatmap_glb_url: string; legend: Record<string, string>;
  store_view: { recommended_size: string | null; confidence: number; zones: { zone: string; verdict: string }[] };
}
interface EventRow { id: string; title: string; starts_at: string; circle: string; dress_code: string; seated_minutes: number }
interface Pick { item_id: string; name: string; slot: string; reasons: string[]; fit_notes: string[]; fit_state: string }
interface Occasion { outfit: Pick[]; alternatives: Record<string, Pick[]>; notes: string[] }
interface OrderItem {
  title: string; size: string; guarantee: boolean;
  unboxing: { measured_cm: Record<string, number>; label_chart_cm: Record<string, number | null>; overlay_image_url: string } | null;
  return: { status: string; route: string; reason: string } | null;
}
interface Order { order_id: string; items: OrderItem[] }
interface GiftView { recommended_size: string | null; confidence: number | null; status: string }

const label = (z: string) => z.replace(/_/g, ' ');
const VERDICT_STYLE: Record<string, string> = {
  ideal: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
  tight: 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300',
  loose: 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300',
  short: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
  long: 'bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-300',
};
const ROUTE_TEXT: Record<string, string> = {
  auto_exchange: 'Label did not match the garment: exchanged automatically',
  auto_exchange_next_size: 'Size chart was off for this body: next size sent automatically',
  human_review: 'A person at the store looks at it',
  merchant_policy: 'No fit guarantee: the store’s own return policy applies',
};

const Chip: React.FC<{ v: string }> = ({ v }) => (
  <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${VERDICT_STYLE[v] ?? 'bg-slate-100 text-slate-700'}`}>{v}</span>
);
const Sim: React.FC<{ children?: React.ReactNode }> = ({ children }) => (
  <span className="inline-block text-[11px] font-bold uppercase tracking-wide px-2 py-0.5 rounded bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300">
    {children ?? 'Simulated'}
  </span>
);
const Card: React.FC<{ n: number; title: string; done: boolean; open: boolean; onOpen: () => void; children: React.ReactNode }> = ({ n, title, done, open, onOpen, children }) => (
  <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
    <button type="button" onClick={onOpen} className="w-full flex items-center gap-3 p-4 text-left" aria-expanded={open}>
      <span className={`h-8 w-8 rounded-full flex items-center justify-center text-sm font-bold ${done ? 'bg-emerald-600 text-white' : 'bg-sky-600 text-white'}`}>
        {done ? <CheckCircle2 className="w-5 h-5" /> : n}
      </span>
      <span className="font-bold text-lg flex-1">{title}</span>
      <ChevronRight className={`w-5 h-5 transition-transform ${open ? 'rotate-90' : ''}`} />
    </button>
    {open && <div className="px-4 pb-5 space-y-4">{children}</div>}
  </section>
);

export const JudgePath: React.FC = () => {
  const session = currentSession();
  const [open, setOpen] = useState(1);
  const [done, setDone] = useState<Set<number>>(new Set());
  const [error, setError] = useState('');
  const finish = (n: number, next?: number) => { setDone((d) => new Set(d).add(n)); if (next) setOpen(next); };
  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : 'Something went wrong.'), []);

  const [twin, setTwin] = useState<Twin | null>(null);
  const [products, setProducts] = useState<ProductCard[]>([]);
  const [fit, setFit] = useState<Fit | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [occasion, setOccasion] = useState<{ title: string; data: Occasion } | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [giftView, setGiftView] = useState<GiftView | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!session) return;
    Promise.all([api<Twin>('/v1/me/twin'), api<ProductCard[]>('/v1/me/products'), api<EventRow[]>('/v1/me/events'), api<Order[]>('/v1/me/orders')])
      .then(([t, p, e, o]) => { setTwin(t); setProducts(p); setEvents(e); setOrders(o); })
      .catch(fail);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!session) {
    return (
      <div className="max-w-xl mx-auto px-4 py-16 text-center space-y-4">
        <h1 className="text-2xl font-bold">Start the demo first</h1>
        <p className="text-slate-600 dark:text-slate-400">The demo gives you your own sandbox with a simulated person, closet and orders.</p>
        <Link to="/" className="inline-block px-6 py-3 rounded-xl bg-sky-600 text-white font-semibold">Back to the start</Link>
      </div>
    );
  }

  const checkProduct = (ref: string) => { setBusy(true); api<Fit>(`/v1/me/fits/${ref}`).then((f) => { setFit(f); finish(2); }).catch(fail).finally(() => setBusy(false)); };
  const testEvent = (e: EventRow) => { setBusy(true); api<Occasion>('/v1/me/occasion-test', { method: 'POST', body: { event_id: e.id } }).then((d) => { setOccasion({ title: e.title, data: d }); finish(3); }).catch(fail).finally(() => setBusy(false)); };
  const approveGift = () => {
    setBusy(true);
    api(`/v1/gifts/${session.gift_id}/approve`, { method: 'POST' })
      .then(() => api<GiftView>(`/v1/me/gifts/${session.gift_id}/giver-view`)).then((v) => { setGiftView(v); finish(4); }).catch(fail).finally(() => setBusy(false));
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-black">Your demo, in five steps</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          <Sim /> The person, closet, calendar and orders below are made up for this demo. Your sandbox is private to you and is deleted automatically.
        </p>
      </header>
      {error && <div role="alert" className="rounded-xl bg-red-50 dark:bg-red-950 text-red-800 dark:text-red-200 p-3 text-sm">{error}</div>}

      <Card n={1} title="See your measured twin" done={done.has(1)} open={open === 1} onOpen={() => setOpen(1)}>
        {!twin ? <Loader2 className="w-6 h-6 animate-spin" /> : (
          <>
            <div className="h-72 rounded-xl bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <model-viewer src={twin.glb_url} alt="Stand-in 3D body for the demo persona" camera-controls auto-rotate shadow-intensity="1" style={{ width: '100%', height: '100%' }} />
            </div>
            <p className="text-xs"><Sim>Simulated</Sim> A stand-in mannequin sized from simulated measurements. It is not an SMPL scan and not a real person.</p>
            <dl className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-sm">
              {Object.entries(twin.zones).map(([z, v]) => (
                <div key={z} className="rounded-lg bg-slate-50 dark:bg-slate-800 p-2"><dt className="text-xs text-slate-500">{label(z)}</dt><dd className="font-semibold">{v.value_cm} cm</dd></div>
              ))}
            </dl>
            <p className="text-sm flex items-start gap-2"><ShieldCheck className="w-4 h-4 mt-0.5 text-emerald-600" />{twin.privacy}</p>
            <button className="px-4 py-2 rounded-lg bg-sky-600 text-white font-semibold" onClick={() => finish(1, 2)}>Next: check a product</button>
          </>
        )}
      </Card>

      <Card n={2} title="Check a product" done={done.has(2)} open={open === 2} onOpen={() => setOpen(2)}>
        <div className="flex flex-wrap gap-2">
          {products.map((p) => (
            <button key={p.ref} disabled={busy} onClick={() => checkProduct(p.ref)} className={`px-3 py-1.5 rounded-lg border text-sm ${fit?.title === p.title ? 'bg-sky-600 text-white border-sky-600' : 'border-slate-300 dark:border-slate-700'}`}>
              {p.title}
            </button>
          ))}
        </div>
        {fit && (
          <div className="space-y-3">
            <p className="font-semibold">
              {fit.recommended_size ? `Recommended size ${fit.recommended_size}, confidence ${Math.round(fit.confidence * 100)}%` : `No size in this chart fits well. Closest: ${fit.shown_size}`}
            </p>
            <div className="h-64 rounded-xl bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <model-viewer src={fit.heatmap_glb_url} alt={`Fit heatmap for ${fit.title}`} camera-controls auto-rotate style={{ width: '100%', height: '100%' }} />
            </div>
            <div className="flex flex-wrap gap-2 text-xs">{Object.entries(fit.legend).map(([k, v]) => <span key={k}><Chip v={k} /> {v}</span>)}</div>
            <ul className="text-sm divide-y divide-slate-100 dark:divide-slate-800">
              {fit.zones.map((z) => (
                <li key={z.zone} className="py-1.5 flex items-center justify-between gap-2">
                  <span className="capitalize">{label(z.zone)}{z.hard ? ' (hard to alter)' : ''}</span>
                  <span className="flex items-center gap-2"><Chip v={z.verdict} />{z.verdict_after_wash !== z.verdict && <>after a wash <Chip v={z.verdict_after_wash} /></>}{z.pose_strain.length > 0 && <span className="text-xs text-amber-700">strain when {z.pose_strain.join(', ')}</span>}</span>
                </li>
              ))}
            </ul>
            {[...fit.warnings, ...fit.alterations, ...fit.review_notes].map((t, i) => <p key={i} className="text-sm text-slate-600 dark:text-slate-400">• {t}</p>)}
            <div className="rounded-xl border border-dashed border-slate-300 dark:border-slate-700 p-3 text-sm">
              <p className="font-semibold">What the store sees (verdicts only, no measurements)</p>
              <p>Size {fit.store_view.recommended_size ?? 'none'}, confidence {Math.round(fit.store_view.confidence * 100)}%: {fit.store_view.zones.map((z) => `${label(z.zone)} ${z.verdict}`).join(', ') || 'no verdicts'}</p>
            </div>
            <button className="px-4 py-2 rounded-lg bg-sky-600 text-white font-semibold" onClick={() => setOpen(3)}>Next: plan an occasion</button>
          </div>
        )}
      </Card>

      <Card n={3} title="Test an outfit for an event" done={done.has(3)} open={open === 3} onOpen={() => setOpen(3)}>
        <ul className="space-y-2">
          {events.map((e) => (
            <li key={e.id} className="flex items-center justify-between gap-2 rounded-lg bg-slate-50 dark:bg-slate-800 p-2 text-sm">
              <span><b>{e.title}</b> · {new Date(e.starts_at).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })} · <span className="px-1.5 rounded bg-slate-200 dark:bg-slate-700 text-xs">{e.circle}</span></span>
              <button disabled={busy} onClick={() => testEvent(e)} className="px-3 py-1 rounded-lg bg-sky-600 text-white text-xs font-semibold">Test outfit</button>
            </li>
          ))}
        </ul>
        {occasion && (
          <div className="space-y-2">
            <p className="font-semibold">Outfit for {occasion.title}</p>
            {occasion.data.outfit.map((p) => (
              <div key={p.item_id} className="rounded-lg border border-slate-200 dark:border-slate-800 p-2 text-sm">
                <p><b>{p.name}</b> <span className="text-xs text-slate-500">({p.slot})</span> {p.fit_state === 'measured_watch' && <span className="text-amber-700 text-xs font-semibold">check the fit note</span>}</p>
                {[...p.reasons, ...p.fit_notes].map((r, i) => <p key={i} className="text-slate-600 dark:text-slate-400">• {r}</p>)}
              </div>
            ))}
            {occasion.data.notes.map((n, i) => <p key={i} className="text-xs text-slate-500">{n}</p>)}
            <button className="px-4 py-2 rounded-lg bg-sky-600 text-white font-semibold" onClick={() => setOpen(4)}>Next: orders and gifts</button>
          </div>
        )}
      </Card>

      <Card n={4} title="Unboxing and a gift request" done={done.has(4)} open={open === 4} onOpen={() => setOpen(4)}>
        <p className="text-sm"><Sim /> Orders and unboxing measurements are simulated. Each return follows a different route.</p>
        {orders.flatMap((o) => o.items.map((it) => (
          <div key={o.order_id + it.title} className="rounded-lg border border-slate-200 dark:border-slate-800 p-3 text-sm space-y-1">
            <p><b>{it.title}</b>, size {it.size} {it.guarantee && <span className="text-emerald-700 text-xs font-semibold">fit guarantee</span>}</p>
            {it.unboxing && (
              <p className="text-slate-600 dark:text-slate-400">
                Measured on arrival vs. the size chart: {Object.entries(it.unboxing.measured_cm).map(([z, v]) => `${label(z)} ${v} vs ${it.unboxing!.label_chart_cm[z] ?? '?'} cm`).join(' · ')}
              </p>
            )}
            {it.return && <p>Return: {ROUTE_TEXT[it.return.route] ?? it.return.route}</p>}
          </div>
        )))}
        <div className="rounded-lg bg-slate-50 dark:bg-slate-800 p-3 text-sm space-y-2">
          <p><b>A friend asks for your size</b> for a Field Canvas Utility Jacket <Sim /></p>
          {giftView ? (
            <p>Approved. Your friend can see only: size {giftView.recommended_size}, confidence {Math.round((giftView.confidence ?? 0) * 100)}%. No measurements, photos or body details.</p>
          ) : (
            <button disabled={busy} onClick={approveGift} className="px-3 py-1.5 rounded-lg bg-sky-600 text-white font-semibold">Share size and confidence only</button>
          )}
        </div>
        {giftView && <button className="px-4 py-2 rounded-lg bg-sky-600 text-white font-semibold" onClick={() => setOpen(5)}>Next: the business side</button>}
      </Card>

      <Card n={5} title="The business side" done={done.has(5)} open={open === 5} onOpen={() => setOpen(5)}>
        <p className="text-sm">A store gets a size and fit verdicts, never your numbers. Try both business views:</p>
        <div className="flex flex-wrap gap-3">
          <a href="/store/shirt.html" target="_blank" rel="noreferrer" onClick={() => finish(5)} className="px-4 py-2 rounded-lg bg-slate-900 text-white font-semibold">Open the store and press “Find my size”</a>
          <Link to="/biz/returns" onClick={() => finish(5)} className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 font-semibold">Open the returns desk</Link>
        </div>
        <p className="text-xs text-slate-500">In the store, the pop-up asks you to approve sharing. Your sandbox is already signed in, so approve and the size appears on the product page.</p>
      </Card>
    </div>
  );
};
