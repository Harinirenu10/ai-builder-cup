import { useCallback, useEffect, useState } from 'react';
import { GoogleAuthProvider, onAuthStateChanged, signInWithPopup, type User } from 'firebase/auth';
import { auth } from './firebase';

// Sign-in for the business desk. VITE_AUTH_MODE=dev (local work and tests only) lets you type a made-up user id;
// otherwise it is Google sign-in through Firebase Auth and the API verifies the ID token.

export const API = ((import.meta.env.VITE_API_URL as string | undefined) ?? '/api').replace(/\/+$/, '');
export const DEV_AUTH = import.meta.env.VITE_AUTH_MODE === 'dev';

export function useDeskAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [devUid, setDevUid] = useState('');
  const [error, setError] = useState('');

  useEffect(() => onAuthStateChanged(auth, (u) => setUser(u)), []);

  const signedIn = DEV_AUTH ? devUid !== '' : !!user;

  const signIn = useCallback(async (uid?: string) => {
    setError('');
    if (DEV_AUTH) {
      setDevUid((uid ?? '').trim());
      return;
    }
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
    } catch {
      setError('Sign-in did not finish. Please try again.');
    }
  }, []);

  const bearer = useCallback(async (): Promise<string> => (DEV_AUTH ? `dev:${devUid}` : await user!.getIdToken()), [devUid, user]);

  return { signedIn, signIn, bearer, error };
}
