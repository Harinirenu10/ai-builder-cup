import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { GoogleAuthProvider, onAuthStateChanged, signInWithPopup, type User } from 'firebase/auth';
import { ShieldCheck } from 'lucide-react';
import { auth } from '../lib/firebase';

// The consent popup a store's size widget opens: /app/connect?merchant=<publishable key>&origin=<store origin>.
// Sign in, approve, then post a signed consent token back to the opener, and only to the origin the API accepted.

const API = ((import.meta.env.VITE_API_URL as string | undefined) ?? '/api').replace(/\/+$/, '');
const DEV_AUTH = import.meta.env.VITE_AUTH_MODE === 'dev'; // local work and tests only: sign in as a made-up user id
const MSG_OK = 'measured-fit:consent';
const MSG_NO = 'measured-fit:denied';

type Phase = 'checking' | 'rejected' | 'signin' | 'ready' | 'sending' | 'done' | 'error';

export const ConnectPage: React.FC = () => {
  const [params] = useSearchParams();
  const merchant = params.get('merchant') ?? '';
  const originParam = params.get('origin') ?? '';
  const origin = useMemo(() => {
    try {
      const u = new URL(originParam);
      return u.protocol === 'http:' || u.protocol === 'https:' ? u.origin : '';
    } catch {
      return '';
    }
  }, [originParam]);

  const [phase, setPhase] = useState<Phase>('checking');
  const [storeName, setStoreName] = useState('');
  const [user, setUser] = useState<User | null>(null);
  const [devUid, setDevUid] = useState('demo-shopper');
  const [devSignedIn, setDevSignedIn] = useState(false);
  const [message, setMessage] = useState('');

  // 1. Ask the API who is asking and whether the opener is a registered origin, before showing anything to approve.
  useEffect(() => {
    if (!merchant || !origin || !window.opener) {
      setMessage(!window.opener ? 'Open this window from a store’s “Find my size” button.' : 'This link is incomplete.');
      setPhase('rejected');
      return;
    }
    fetch(`${API}/v1/merchants/lookup?key=${encodeURIComponent(merchant)}&origin=${encodeURIComponent(origin)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('unknown'))))
      .then((m: { name: string; origin_allowed: boolean }) => {
        if (!m.origin_allowed) {
          setMessage('This store is not registered with Measured Fit, so nothing will be shared with it.');
          setPhase('rejected');
          return;
        }
        setStoreName(m.name);
        setPhase('signin');
      })
      .catch(() => {
        setMessage('We could not verify this store. Nothing was shared.');
        setPhase('rejected');
      });
  }, [merchant, origin]);

  useEffect(() => onAuthStateChanged(auth, (u) => setUser(u)), []);
  const signedIn = DEV_AUTH ? devSignedIn : !!user;
  useEffect(() => {
    if (phase === 'signin' && signedIn) setPhase('ready');
  }, [phase, signedIn]);

  const bearer = async (): Promise<string> => (DEV_AUTH ? `dev:${devUid.trim()}` : await user!.getIdToken());

  const signIn = async () => {
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
    } catch {
      setMessage('Sign-in did not finish. Please try again.');
    }
  };

  const approve = async () => {
    setPhase('sending');
    try {
      const res = await fetch(`${API}/v1/consents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${await bearer()}` },
        body: JSON.stringify({ merchant, origin }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const c: { token: string; expires_at: string } = await res.json();
      window.opener?.postMessage({ type: MSG_OK, token: c.token, expires_at: c.expires_at }, origin); // exact origin, never '*'
      setPhase('done');
      window.setTimeout(() => window.close(), 400);
    } catch {
      setMessage('We could not save your approval. Nothing was shared. Please try again.');
      setPhase('error');
    }
  };

  const deny = () => {
    window.opener?.postMessage({ type: MSG_NO }, origin);
    window.close();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-4 text-slate-900 dark:text-slate-100">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 space-y-5 shadow-lg" data-testid="connect-card">
        <div className="flex items-center space-x-2 text-sky-600 dark:text-sky-400">
          <ShieldCheck className="w-5 h-5" />
          <span className="font-semibold">Measured Fit</span>
        </div>

        {phase === 'checking' && <p>Checking this store…</p>}

        {phase === 'rejected' && (
          <p role="alert" data-testid="connect-rejected">{message}</p>
        )}

        {(phase === 'signin' || phase === 'ready' || phase === 'sending' || phase === 'error' || phase === 'done') && (
          <>
            <h1 className="text-xl font-bold">Share your size with {storeName}?</h1>
            <ul className="text-sm space-y-2 text-slate-600 dark:text-slate-300 list-disc pl-5">
              <li>{storeName} will learn which size fits you for the garments you look at, and how confident we are.</li>
              <li>It will not receive your measurements, photos or body shape.</li>
              <li>You can turn this off any time under My twin. Turning it off stops all later answers.</li>
              <li>The permission lasts 30 days.</li>
            </ul>

            {!signedIn && (
              <div className="space-y-3">
                {DEV_AUTH ? (
                  <>
                    <label className="block text-sm" htmlFor="dev-uid">Dev sign-in (test mode only)</label>
                    <input id="dev-uid" value={devUid} onChange={(e) => setDevUid(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2" />
                    <button type="button" onClick={() => setDevSignedIn(!!devUid.trim())}
                      className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in</button>
                  </>
                ) : (
                  <button type="button" onClick={signIn}
                    className="w-full rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 py-2 font-medium">Sign in with Google</button>
                )}
              </div>
            )}

            {signedIn && phase !== 'done' && (
              <div className="flex gap-3">
                <button type="button" onClick={approve} disabled={phase === 'sending'}
                  className="flex-1 rounded-lg bg-sky-600 text-white py-2 font-medium disabled:opacity-60">{phase === 'sending' ? 'Sharing…' : 'Approve'}</button>
                <button type="button" onClick={deny} className="flex-1 rounded-lg border border-slate-300 dark:border-slate-700 py-2 font-medium">Not now</button>
              </div>
            )}

            {phase === 'done' && <p role="status" data-testid="connect-done">Done. You can close this window.</p>}
            {phase === 'error' && <p role="alert" className="text-sm text-red-600">{message}</p>}
          </>
        )}
      </div>
    </div>
  );
};

export default ConnectPage;
