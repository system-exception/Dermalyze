import type { ClassResult } from './types';
import { supabase } from './supabase';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

interface ClassifyResponse {
  classes: ClassResult[];
  gradcam_image?: string | null;  // Base64-encoded PNG heatmap overlay
}

export interface ClassifyResult {
  classes: ClassResult[];
  gradcamImage?: string;  // Base64 data URL for display
}

const getApiBaseUrl = (): string => {
  const configured = (import.meta.env as ImportMetaEnv & { VITE_API_URL?: string }).VITE_API_URL;
  if (configured && configured.trim().length > 0) {
    return configured.replace(/\/$/, '');
  }
  return '/api';
};

const API_BASE_URL = getApiBaseUrl();

const assertValidResponse = (payload: unknown): ClassifyResponse => {
  if (!payload || typeof payload !== 'object' || !('classes' in payload)) {
    throw new Error('Invalid API response: missing classes field.');
  }

  const maybeClasses = (payload as { classes: unknown }).classes;
  if (!Array.isArray(maybeClasses)) {
    throw new Error('Invalid API response: classes must be an array.');
  }

  return payload as ClassifyResponse;
};

export const classifyImage = async (
  imageDataUrl: string,
  includeGradcam: boolean = true
): Promise<ClassifyResult> => {
  // Get current Supabase session and check expiry
  let { data: { session }, error: sessionError } = await supabase.auth.getSession();

  if (sessionError || !session?.access_token) {
    throw new ApiError('Authentication required. Please log in again.', 401);
  }

  // Check if token will expire within next 60 seconds, refresh if needed
  const expiresAt = session.expires_at ? session.expires_at * 1000 : 0; // Convert to ms
  const now = Date.now();
  const expiresInMs = expiresAt - now;

  if (expiresInMs < 60_000) { // Less than 60 seconds remaining
    const { data: refreshData, error: refreshError } = await supabase.auth.refreshSession();
    if (refreshError || !refreshData.session?.access_token) {
      throw new ApiError('Session expired. Please log in again.', 401);
    }
    session = refreshData.session;
  }

  const imageBlob = await (await fetch(imageDataUrl)).blob();
  const imageExt = imageBlob.type === 'image/png' ? 'png' : 'jpg';

  const formData = new FormData();
  formData.append('file', imageBlob, `lesion.${imageExt}`);

  // Add include_gradcam query parameter
  const url = new URL(`${API_BASE_URL}/classify`, window.location.origin);
  url.searchParams.set('include_gradcam', includeGradcam.toString());

  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${session.access_token}`,
    },
    body: formData,
  });

  const json = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      json && typeof json === 'object' && 'detail' in json && typeof json.detail === 'string'
        ? json.detail
        : `Classification failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }

  const parsed = assertValidResponse(json);
  const sortedClasses = [...parsed.classes].sort((a, b) => b.score - a.score);

  // Convert base64 to data URL if gradcam_image is present
  const gradcamImage = parsed.gradcam_image
    ? `data:image/png;base64,${parsed.gradcam_image}`
    : undefined;

  return {
    classes: sortedClasses,
    gradcamImage,
  };
};
