import { useQuery } from '@tanstack/react-query';
import { apiClient } from './api';
import { describeJobStage, isPollingStatus, isTerminalStatus, type JobStatusType } from './jobStatus';

export type { JobStatusType };

export interface TwinJobDocument {
  id: string;
  user_id: string;
  status: JobStatusType;
  height_cm: number;
  gender: string;
  twin_id?: string;
  final_mean_iou?: number;
  error_code?: string;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export function useJobStatus(jobId: string | null) {
  const {
    data: job,
    isLoading,
    error,
    refetch,
  } = useQuery<TwinJobDocument>({
    queryKey: ['job-status', jobId],
    queryFn: async () => {
      if (!jobId) throw new Error('No job ID provided');
      return await apiClient.get<TwinJobDocument>(`/api/twin/jobs/${jobId}`);
    },
    enabled: !!jobId,
    refetchInterval: (query) => (isPollingStatus(query.state.data?.status) ? 1500 : false),
    retry: 3,
    retryDelay: 1000,
  });

  const status = job?.status;
  const { stage, progressPercent } = describeJobStage(status, job?.error_message);

  return {
    job,
    status,
    stage,
    progressPercent,
    isTerminal: isTerminalStatus(status),
    isLoading,
    error,
    refetch,
  };
}
