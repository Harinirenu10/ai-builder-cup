import { useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, ApiError } from './api';
import { useAuth } from './auth';

export interface CustomerProgressData {
  user_id?: string;
  active_tab: 'twin' | 'capture' | 'measurements';
  height_cm: string;
  gender: 'neutral' | 'female' | 'male';
  age_18_confirmed: boolean;
  opt_in_keep_front_photo: boolean;
  captured_views: string[];
  last_job_id: string | null;
  updated_at?: string;
}

const DEFAULT_PROGRESS: CustomerProgressData = {
  active_tab: 'twin',
  height_cm: '175.0',
  gender: 'neutral',
  age_18_confirmed: false,
  opt_in_keep_front_photo: false,
  captured_views: [],
  last_job_id: null,
};

const getStorageKey = (uid?: string) => `omni_progress_${uid || 'guest'}`;

export function useCustomerProgress() {
  const queryClient = useQueryClient();
  const { user, signedIn } = useAuth();
  const uid = user?.uid;

  // 1. Fetch saved progress from backend via TanStack Query
  const {
    data: progress = DEFAULT_PROGRESS,
    isLoading,
    refetch,
  } = useQuery<CustomerProgressData>({
    queryKey: ['progress', uid],
    queryFn: async () => {
      try {
        const remote = await apiClient.get<CustomerProgressData>('/v1/me/progress');
        // Cache to local storage
        try {
          localStorage.setItem(getStorageKey(uid), JSON.stringify(remote));
        } catch {
          // ignore
        }
        return remote;
      } catch (err: any) {
        // Fallback to local storage if offline or network error
        try {
          const cached = localStorage.getItem(getStorageKey(uid));
          if (cached) return JSON.parse(cached);
        } catch {
          // ignore
        }
        if (err instanceof ApiError && err.status === 401) {
          return DEFAULT_PROGRESS;
        }
        return DEFAULT_PROGRESS;
      }
    },
    enabled: !!signedIn,
    staleTime: 1000 * 60, // 1 minute
  });

  // 2. Mutation to persist updated progress state
  const saveMutation = useMutation({
    mutationFn: async (updated: CustomerProgressData) => {
      // Save locally first for instant response
      try {
        localStorage.setItem(getStorageKey(uid), JSON.stringify(updated));
      } catch {
        // ignore
      }
      if (signedIn) {
        return await apiClient.put<CustomerProgressData>('/v1/me/progress', updated);
      }
      return updated;
    },
    onSuccess: (saved) => {
      queryClient.setQueryData(['progress', uid], saved);
    },
  });

  // Helper to save partial progress
  const updateProgress = useCallback(
    (partial: Partial<CustomerProgressData>) => {
      const merged: CustomerProgressData = {
        ...progress,
        ...partial,
        updated_at: new Date().toISOString(),
      };
      saveMutation.mutate(merged);
    },
    [progress, saveMutation]
  );

  return {
    progress,
    isLoading,
    refetch,
    updateProgress,
    isSaving: saveMutation.isPending,
  };
}
