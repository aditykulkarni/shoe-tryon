import { test } from 'node:test';
import assert from 'node:assert/strict';
import { coverMapper, footAnchor } from '../src/lib/footPose.js';

const lm = (x, y, z = 0, visibility = 0.9) => ({ x, y, z, visibility });

function pose({ heel, toe, ankle, world }) {
  const landmarks = Array.from({ length: 33 }, () => lm(0, 0, 0, 0));
  const worldLandmarks = Array.from({ length: 33 }, () => lm(0, 0, 0, 0));
  [landmarks[30], landmarks[32], landmarks[28]] = [heel, toe, ankle];
  [worldLandmarks[30], worldLandmarks[32], worldLandmarks[28]] = world;
  return { landmarks, worldLandmarks };
}

test('cover mapping crops the long side evenly', () => {
  const map = coverMapper(1280, 720, 390, 844); // landscape video on a portrait phone
  const [x, y] = map(0.5, 0.5);
  assert.ok(Math.abs(x - 195) < 1e-9 && Math.abs(y - 422) < 1e-9);
});

test('foot lying flat and sideways gives an orthonormal frame toward the toe', () => {
  // World: heel at origin, toe 0.25 m to the right, ankle 0.08 m up (y down in MediaPipe).
  const { landmarks, worldLandmarks } = pose({
    heel: lm(0.4, 0.8), toe: lm(0.6, 0.8), ankle: lm(0.41, 0.72),
    world: [lm(0, 0, 0), lm(0.25, 0, 0), lm(0.01, -0.08, 0)],
  });
  const a = footAnchor(landmarks, worldLandmarks, 'right', (x, y) => [x * 1000, y * 1000]);
  const { right, up, forward } = a.basis;
  assert.deepEqual(forward.map((v) => +v.toFixed(6)), [1, 0, 0]);
  assert.deepEqual(up.map((v) => +v.toFixed(6)), [0, 1, 0]);
  assert.deepEqual(right.map((v) => +v.toFixed(6)), [0, 0, -1]);
  assert.ok(Math.abs(a.pixelsPerFootLength - 200) < 1e-9);
});

test('hidden foot returns null', () => {
  const { landmarks, worldLandmarks } = pose({
    heel: lm(0.4, 0.8, 0, 0.1), toe: lm(0.6, 0.8), ankle: lm(0.41, 0.72),
    world: [lm(0, 0, 0), lm(0.25, 0, 0), lm(0, -0.08, 0)],
  });
  assert.equal(footAnchor(landmarks, worldLandmarks, 'right', (x, y) => [x, y]), null);
});
