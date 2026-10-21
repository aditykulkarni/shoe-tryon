// Remembers the last measurement on this phone. Storage can be blocked
// (private mode), so every access is guarded.
const KEY = 'ar-shoe-tryon:measurement';

export function saveMeasurement(m) {
  try {
    localStorage.setItem(KEY, JSON.stringify(m));
  } catch {
    // Not fatal: try-on falls back to the default foot length.
  }
}

export function loadMeasurement() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) ?? null;
  } catch {
    return null;
  }
}
