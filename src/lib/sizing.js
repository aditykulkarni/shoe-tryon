// Paper sizes in millimetres.
export const PAPERS = {
  a4: { label: 'A4', short: 210, long: 297 },
  letter: { label: 'US Letter', short: 215.9, long: 279.4 },
};

const roundHalf = (n) => Math.round(n * 2) / 2;

// Rule-of-thumb conversions from foot length. Brand size charts should
// replace these before anyone buys from them.
export function sizesFromLength(lengthMm) {
  const inches = lengthMm / 25.4;
  const usMen = roundHalf(3 * inches - 22);
  return {
    mondopoint: Math.round(lengthMm),
    // Paris point is 2/3 cm; the shoe last runs about 15 mm longer than the foot.
    eu: roundHalf(((lengthMm + 15) * 1.5) / 10),
    usMen,
    usWomen: usMen + 1.5,
    uk: usMen - 0.5,
  };
}

// Width relative to length, bucketed into the usual width letters.
export function widthClass(lengthMm, widthMm) {
  const ratio = widthMm / lengthMm;
  if (ratio < 0.36) return 'narrow';
  if (ratio < 0.4) return 'standard';
  return 'wide';
}
