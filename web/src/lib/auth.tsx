import React, { createContext, useContext, useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { GoogleAuthProvider, onAuthStateChanged, signInWithPopup, signOut as fbSignOut, type User } from 'firebase/auth';
import { auth } from './firebase';
import { apiClient } from './api';

export interface AuthUser {
  uid: string;
  email?: string | null;
  displayName?: string | null;
  isDemo: boolean;
}

export interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  signedIn: boolean;
  isLoading: boolean;
  error: string | null;
  signInWithDemo: (uid?: string) => void;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  getToken: () => Promise<string | null>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

const DEMO_UID_KEY = 'omni_demo_uid';
const TOKEN_KEY = 'omni_token';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clearError = useCallback(() => setError(null), []);

  // Demo sign-in: sets demo shopper and dev: token compatible with backend AUTH_MODE=dev and seed.py
  const signInWithDemo = useCallback((uid: string = 'demo-shopper') => {
    const cleanUid = uid.trim() || 'demo-shopper';
    const demoToken = `dev:${cleanUid}`;
    const demoUser: AuthUser = {
      uid: cleanUid,
      email: `${cleanUid}@demo.local`,
      displayName: cleanUid === 'demo-shopper' ? 'Demo Shopper' : cleanUid,
      isDemo: true,
    };
    try {
      localStorage.setItem(DEMO_UID_KEY, cleanUid);
      localStorage.setItem(TOKEN_KEY, demoToken);
    } catch {
      // Storage unavailable (e.g. private mode)
    }
    setToken(demoToken);
    setUser(demoUser);
    setError(null);
  }, []);

  // Sign out clears both Firebase auth and demo storage
  const signOut = useCallback(async () => {
    try {
      localStorage.removeItem(DEMO_UID_KEY);
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      // ignore
    }
    setToken(null);
    setUser(null);
    setError(null);
    try {
      await fbSignOut(auth);
    } catch {
      // ignore
    }
  }, []);

  // Real Google Sign-in via Firebase
  const signInWithGoogle = useCallback(async () => {
    setError(null);
    try {
      const provider = new GoogleAuthProvider();
      const credential = await signInWithPopup(auth, provider);
      const fbUser = credential.user;
      const idToken = await fbUser.getIdToken();
      
      try {
        localStorage.removeItem(DEMO_UID_KEY);
        localStorage.setItem(TOKEN_KEY, idToken);
      } catch {
        // ignore
      }

      setToken(idToken);
      setUser({
        uid: fbUser.uid,
        email: fbUser.email,
        displayName: fbUser.displayName,
        isDemo: false,
      });
    } catch (err: any) {
      if (err.code !== 'auth/popup-closed-by-user') {
        setError(err.message || 'Failed to sign in with Google');
      }
    }
  }, []);

  // Get current bearer token (refreshes Firebase token if needed)
  const getToken = useCallback(async (): Promise<string | null> => {
    if (user?.isDemo) {
      return token || `dev:${user.uid}`;
    }
    if (auth.currentUser) {
      try {
        const freshToken = await auth.currentUser.getIdToken();
        setToken(freshToken);
        try {
          localStorage.setItem(TOKEN_KEY, freshToken);
        } catch {
          // ignore
        }
        return freshToken;
      } catch {
        return token;
      }
    }
    return token;
  }, [user, token]);

  // Sync token getter with apiClient. Registered ONCE via a stable wrapper that reads
  // `getTokenRef.current` (updated synchronously during render, every render, below) -- not
  // re-registered in a `useEffect([getToken])` keyed to the latest closure. Passive effects run
  // bottom-up (deepest descendants first), so a child component's own effect -- e.g. a
  // TanStack Query enabled by `signedIn` flipping true on this exact render, which fires in
  // CustomerApp.tsx, a descendant of this provider -- would otherwise run its fetch BEFORE this
  // effect re-registered the fresh getToken, using a stale closure that still saw the
  // pre-sign-in `user`/`token` and sent no Authorization header at all (observed as a real 401
  // on /v1/me/twin and /v1/me/progress immediately after demo sign-in, caught by
  // services/api/tests/test_customer_app_e2e.py). The ref is updated in the render body itself,
  // so it is current by the time ANY effect in the tree runs this render, regardless of order.
  const getTokenRef = useRef(getToken);
  getTokenRef.current = getToken;
  useEffect(() => {
    apiClient.setTokenGetter(() => getTokenRef.current());
  }, []);

  // Listen for unauthorized 401 events from apiClient
  useEffect(() => {
    const handleUnauthorized = () => {
      // If unauthorized, notify but don't disrupt abruptly if offline
      console.warn('Received 401 Unauthorized from API client');
    };
    window.addEventListener('omni:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('omni:unauthorized', handleUnauthorized);
  }, []);

  // Initial authentication restoration
  useEffect(() => {
    let isMounted = true;

    // First check demo session
    let savedDemoUid: string | null = null;
    try {
      savedDemoUid = localStorage.getItem(DEMO_UID_KEY);
    } catch {
      // ignore
    }

    if (savedDemoUid) {
      const demoToken = `dev:${savedDemoUid}`;
      setUser({
        uid: savedDemoUid,
        email: `${savedDemoUid}@demo.local`,
        displayName: savedDemoUid === 'demo-shopper' ? 'Demo Shopper' : savedDemoUid,
        isDemo: true,
      });
      setToken(demoToken);
      setIsLoading(false);
      return;
    }

    // Otherwise check Firebase auth state
    const unsubscribe = onAuthStateChanged(auth, async (fbUser: User | null) => {
      if (!isMounted) return;

      if (fbUser) {
        try {
          const idToken = await fbUser.getIdToken();
          if (isMounted) {
            setToken(idToken);
            setUser({
              uid: fbUser.uid,
              email: fbUser.email,
              displayName: fbUser.displayName,
              isDemo: false,
            });
          }
        } catch {
          if (isMounted) {
            setUser({
              uid: fbUser.uid,
              email: fbUser.email,
              displayName: fbUser.displayName,
              isDemo: false,
            });
          }
        }
      } else {
        // If not in demo mode and no Firebase user, user is null
        if (!localStorage.getItem(DEMO_UID_KEY)) {
          setUser(null);
          setToken(null);
        }
      }
      setIsLoading(false);
    });

    return () => {
      isMounted = false;
      unsubscribe();
    };
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      signedIn: !!user,
      isLoading,
      error,
      signInWithDemo,
      signInWithGoogle,
      signOut,
      getToken,
      clearError,
    }),
    [user, token, isLoading, error, signInWithDemo, signInWithGoogle, signOut, getToken, clearError]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
