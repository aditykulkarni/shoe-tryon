import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyHomography, computeHomography, measureMm, paperHomography } from '../src/lib/homography.js';
import { PAPERS } from '../src/lib/sizing.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('maps each source corner onto its destination', () => {
  const src = [[10, 20], [400, 40], [380, 600], [30, 550]];
  const dst = [[0, 0], [210, 0], [210, 297], [0, 297]];
  const H = computeHomography(src, dst);
  src.forEach((p, i) => {
    const [u, v] = applyHomography(H, p);
    close(u, dst[i][0]);
    close(v, dst[i][1]);
  });
});

test('recovers true distances from a perspective-distorted sheet', () => {
  // Forward-project a known A4 sheet and two foot points through a tilt.
  const tilt = computeHomography(
    [[0, 0], [210, 0], [210, 297], [0, 297]],
    [[120, 80], [880, 140], [960, 1300], [40, 1240]]
  );
  const project = (p) => applyHomography(tilt, p);
  const corners = [[210, 297], [0, 0], [0, 297], [210, 0]].map(project); // shuffled tap order
  const heel = project([105, 10]);
  const toe = project([95, 275]);
  const H = paperHomography(corners, PAPERS.a4);
  close(measureMm(H, heel, toe), Math.hypot(10, 265), 1e-6);
});

test('works when the sheet is photographed sideways', () => {
  const tilt = computeHomography(
    [[0, 0], [210, 0], [210, 297], [0, 297]],
    [[1200, 100], [1250, 800], [100, 760], [60, 140]]
  );
  const project = (p) => applyHomography(tilt, p);
  const corners = [[0, 0], [210, 0], [210, 297], [0, 297]].map(project);
  const H = paperHomography(corners, PAPERS.a4);
  close(measureMm(H, project([20, 50]), project([190, 50])), 170, 1e-6);
});

test('rejects collinear corners', () => {
  assert.throws(() => computeHomography([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]]));
});
