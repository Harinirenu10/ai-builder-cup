export interface ApiErrorDetail {
  code?: string;
  message?: string;
  [key: string]: any;
}

export class ApiError extends Error {
  public status: number;
  public code: string;
  public details?: any;

  constructor(message: string, status: number, code: string = 'api_error', details?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: any;
  timeoutMs?: number;
  token?: string | null;
  params?: Record<string, string | number | boolean | undefined | null>;
}

export type TokenGetter = () => Promise<string | null> | string | null;

export const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env?.VITE_API_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  // In dev server, /api is proxied by Vite, but direct backend fallback is http://localhost:8000
  if (typeof window !== 'undefined' && window.location?.port === '5173') {
    return '';
  }
  return 'http://localhost:8000';
};

export const buildUrl = (path: string, params?: Record<string, string | number | boolean | undefined | null>): string => {
  let url = path;
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    const base = getApiBaseUrl();
    const cleanPath = url.startsWith('/') ? url : `/${url}`;
    url = base ? `${base}${cleanPath}` : cleanPath;
  }
  if (params) {
    const query = new URLSearchParams();
    for (const [key, val] of Object.entries(params)) {
      if (val !== undefined && val !== null) {
        query.append(key, String(val));
      }
    }
    const queryString = query.toString();
    if (queryString) {
      url += (url.includes('?') ? '&' : '?') + queryString;
    }
  }
  return url;
};

export const parseErrorResponse = async (response: Response): Promise<ApiError> => {
  const status = response.status;
  let code = `http_${status}`;
  let message = response.statusText || `Request failed with status ${status}`;
  let details: any = null;

  try {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const data = await response.json();
      details = data;
      if (typeof data === 'string') {
        message = data;
      } else if (data && typeof data === 'object') {
        if (typeof data.detail === 'string') {
          message = data.detail;
        } else if (data.detail && typeof data.detail === 'object') {
          code = data.detail.code || code;
          message = data.detail.message || message;
        } else if (typeof data.message === 'string') {
          message = data.message;
          if (data.code) code = data.code;
        } else if (typeof data.error_message === 'string') {
          message = data.error_message;
          if (data.error_code) code = data.error_code;
        }
      }
    } else {
      const text = await response.text();
      if (text) message = text;
    }
  } catch {
    // If response body parsing fails, stick to default statusText
  }

  if (status === 401) {
    code = 'unauthorized';
  } else if (status === 403) {
    code = 'forbidden';
  } else if (status === 404) {
    code = 'not_found';
  }

  return new ApiError(message, status, code, details);
};

const DEFAULT_TIMEOUT_MS = 15000;

export class ApiClient {
  private tokenGetter: TokenGetter | null = null;
  private defaultTimeoutMs: number = DEFAULT_TIMEOUT_MS;

  public setTokenGetter(getter: TokenGetter | null) {
    this.tokenGetter = getter;
  }

  public setDefaultTimeout(ms: number) {
    this.defaultTimeoutMs = ms;
  }

  public async request<T = any>(path: string, options: RequestOptions = {}): Promise<T> {
    const {
      body,
      timeoutMs = this.defaultTimeoutMs,
      token: explicitToken,
      params,
      headers: customHeaders = {},
      ...fetchOptions
    } = options;

    const url = buildUrl(path, params);

    // Resolve auth token
    let token = explicitToken;
    if (token === undefined) {
      if (this.tokenGetter) {
        token = await this.tokenGetter();
      } else if (typeof localStorage !== 'undefined') {
        token = localStorage.getItem('omni_token');
      }
    }

    const headers = new Headers(customHeaders);
    if (token) {
      headers.set('Authorization', token.startsWith('Bearer ') ? token : `Bearer ${token}`);
    }

    let finalBody: BodyInit | undefined = undefined;
    if (body !== undefined && body !== null) {
      if (body instanceof FormData || body instanceof Blob || typeof body === 'string') {
        finalBody = body;
      } else {
        headers.set('Content-Type', 'application/json');
        finalBody = JSON.stringify(body);
      }
    }

    const controller = new AbortController();
    let timeoutId: any = null;
    let timedOut = false;

    if (timeoutMs > 0) {
      timeoutId = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs);
    }

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers,
        body: finalBody,
        signal: controller.signal,
      });

      if (!response.ok) {
        if (response.status === 401 && typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
          try {
            const event = typeof CustomEvent !== 'undefined'
              ? new CustomEvent('omni:unauthorized', { detail: { path, status: 401 } })
              : ({ type: 'omni:unauthorized', detail: { path, status: 401 } } as any);
            window.dispatchEvent(event);
          } catch {
            // ignore event dispatch errors
          }
        }
        const error = await parseErrorResponse(response);
        throw error;
      }

      // Check content type for parsing
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return (await response.json()) as T;
      }
      if (contentType.includes('text/')) {
        return (await response.text()) as unknown as T;
      }
      if (response.status === 204) {
        return null as unknown as T;
      }
      return (await response.blob()) as unknown as T;
    } catch (err: any) {
      if (timedOut || err.name === 'AbortError') {
        throw new ApiError(`Request timed out after ${timeoutMs}ms`, 408, 'timeout');
      }
      if (err instanceof ApiError) {
        throw err;
      }
      throw new ApiError(err.message || 'Network request failed', 0, 'network_error', err);
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    }
  }

  public get<T = any>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'GET' });
  }

  public post<T = any>(path: string, body?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'POST', body });
  }

  public put<T = any>(path: string, body?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'PUT', body });
  }

  public patch<T = any>(path: string, body?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'PATCH', body });
  }

  public delete<T = any>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'DELETE' });
  }
}

export const apiClient = new ApiClient();

/**
 * Loads a binary resource (e.g. a 3D twin's GLB mesh) as an authenticated Blob through apiClient
 * -- so the Authorization header reaches endpoints that require it, unlike handing a URL
 * straight to an element such as <model-viewer>, which fetches its own `src` with no header --
 * and wraps it in an object URL. The returned `revoke` is idempotent (calling it more than once
 * only revokes the URL once); callers should run it when the resource changes or the consuming
 * component unmounts, so object URLs never accumulate.
 */
export async function loadObjectUrl(path: string, client: ApiClient = apiClient): Promise<{ url: string; revoke: () => void }> {
  const blob = await client.get<Blob>(path);
  const url = URL.createObjectURL(blob);
  let revoked = false;
  return {
    url,
    revoke: () => {
      if (!revoked) {
        revoked = true;
        URL.revokeObjectURL(url);
      }
    },
  };
}
