// Pure job-status logic, kept dependency-free (no react-query, no apiClient) so it can be
// imported directly in Node for tests, the same way api.ts is. useJobStatus.ts wraps this with
// the actual polling query.
//
// Must match fitcore.schemas.JobStatus exactly (services/geometry/worker.py progresses a job
// through checking -> measuring -> fitting -> refining -> done, or to needs_retake/rejected/
// failed). Missing a non-terminal status here would make isPollingStatus/isTerminalStatus wrong,
// which would silently stop polling while the backend is still working.
export type JobStatusType = 'checking' | 'measuring' | 'fitting' | 'refining' | 'done' | 'needs_retake' | 'rejected' | 'failed';

const TERMINAL_STATUSES: readonly JobStatusType[] = ['done', 'needs_retake', 'rejected', 'failed'];
const POLLING_STATUSES: readonly JobStatusType[] = ['checking', 'measuring', 'fitting', 'refining'];

export function isTerminalStatus(status: JobStatusType | undefined): boolean {
  return !!status && TERMINAL_STATUSES.includes(status);
}

export function isPollingStatus(status: JobStatusType | undefined): boolean {
  return !!status && POLLING_STATUSES.includes(status);
}

export interface JobStageInfo {
  stage: string;
  progressPercent: number;
}

export function describeJobStage(status: JobStatusType | undefined, errorMessage?: string | null): JobStageInfo {
  switch (status) {
    case 'checking':
      return { stage: 'Validating 4 orthogonal views with MediaPipe...', progressPercent: 20 };
    case 'measuring':
      return { stage: 'Extracting body landmarks and measurements...', progressPercent: 45 };
    case 'fitting':
      return { stage: 'Optimizing SMPL 3D parametric body shape...', progressPercent: 65 };
    case 'refining':
      return { stage: 'Refining 3D fit with multimodal critic review...', progressPercent: 85 };
    case 'done':
      return { stage: '3D Body Twin generation complete!', progressPercent: 100 };
    case 'needs_retake':
      return { stage: 'Photos require retake: ' + (errorMessage || 'Please follow pose guidelines.'), progressPercent: 0 };
    case 'rejected':
    case 'failed':
      return { stage: 'Processing halted: ' + (errorMessage || 'Validation rejected.'), progressPercent: 0 };
    default:
      return { stage: 'Preparing submission...', progressPercent: 10 };
  }
}
