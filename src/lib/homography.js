// Plane-to-plane homography from four point pairs, used to turn a tilted
// photo of a sheet of paper into a flat, millimetre-scaled coordinate frame.

// Solves A x = b with Gaussian elimination and partial pivoting.
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) throw new Error('Degenerate points: homography is singular');
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// Returns the 3x3 matrix (row-major, length 9) mapping each src[i] to dst[i].
export function computeHomography(src, dst) {
  if (src.length !== 4 || dst.length !== 4) throw new Error('Need exactly four point pairs');
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  return [...solve(A, b), 1];
}

export function applyHomography(H, [x, y]) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Orders four tapped corners into a cycle and pairs them with the paper's
// rectangle so the short sides map to the paper's short edge. Tap order
// does not matter.
export function paperHomography(corners, paper) {
  const cx = corners.reduce((s, p) => s + p[0], 0) / 4;
  const cy = corners.reduce((s, p) => s + p[1], 0) / 4;
  let ring = [...corners].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)
  );
  const sidesA = dist(ring[0], ring[1]) + dist(ring[2], ring[3]);
  const sidesB = dist(ring[1], ring[2]) + dist(ring[3], ring[0]);
  if (sidesA > sidesB) ring = [ring[1], ring[2], ring[3], ring[0]];
  const { short, long } = paper;
  const rect = [[0, 0], [short, 0], [short, long], [0, long]];
  return computeHomography(ring, rect);
}

// Distance in millimetres between two image points, through the paper plane.
export function measureMm(H, a, b) {
  return dist(applyHomography(H, a), applyHomography(H, b));
}
