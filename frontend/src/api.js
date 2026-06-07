/** API base URL from Vercel env; empty in local dev (Vite proxy handles /predict). */
const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export function apiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}
