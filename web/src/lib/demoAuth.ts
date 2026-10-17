import { signInWithCustomToken } from 'firebase/auth';
import { auth } from './firebase';

// The judge's sandbox: POST /v1/demo/session clones the seeded persona into a fresh user and returns a sign-in token.
// With a real API that is a Firebase custom token; a local API in AUTH_MODE=dev returns a `dev:<uid>` bearer instead.

export const API = ((import.meta.env.VITE_API_URL as string | undefined) ?? '/api').replace(/\/+$/, '');

export interface DemoSession {
  uid: string;
  custom_token: string;
  auth_mode: 'dev' | 'firebase';
  merchant_id: string;
  gift_id: string;
  expires_at: string;
}

const KEY = 'measured-fit:demo';

export function savedSession(): DemoSession | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as DemoSession) : null;
  } catch {
    return null;
  }
}

export async function startDemoSession(): Promise<DemoSession> {
  const r = await fetch(`${API}/v1/demo/session`, { method: 'POST' });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.detail?.message ?? `The demo could not start (${r.status}).`);
  }
  const s = (await r.json()) as DemoSession;
  if (s.auth_mode === 'firebase') await signInWithCustomToken(auth, s.custom_token);
  try {
    sessionStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* private window: the page keeps the session in memory only */
  }
  memory = s;
  return s;
}

let memory: DemoSession | null = null;

export function currentSession(): DemoSession | null {
  return memory ?? savedSession();
}

export async function bearer(): Promise<string> {
  const s = currentSession();
  if (s?.auth_mode === 'dev') return s.custom_token;
  if (!auth.currentUser) throw new Error('Please start the demo again.');
  return auth.currentUser.getIdToken();
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: init.method ?? 'GET',
    headers: { Authorization: `Bearer ${await bearer()}`, ...(init.body ? { 'Content-Type': 'application/json' } : {}) },
    body: init.body ? JSON.stringify(init.body) : undefined,
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.detail?.message ?? `Request failed (${r.status}).`);
  }
  return (await r.json()) as T;
}
