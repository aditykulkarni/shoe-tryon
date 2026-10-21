// Turns MediaPipe pose landmarks into a per-foot anchor: where the heel is
// on screen, which way the toe points, and how big the foot looks.

// BlazePose landmark indices.
export const FOOT_LANDMARKS = {
  left: { ankle: 27, heel: 29, toe: 31 },
  right: { ankle: 28, heel: 30, toe: 32 },
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// MediaPipe world space is x right, y down, z away from the camera;
// three.js is x right, y up, z toward the camera.
const toThree = (p) => [p.x, -p.y, -p.z];

// Maps normalised video coordinates to screen pixels for a video drawn with
// object-fit: cover.
export function coverMapper(videoW, videoH, viewW, viewH) {
  const scale = Math.max(viewW / videoW, viewH / videoH);
  const offX = (viewW - videoW * scale) / 2;
  const offY = (viewH - videoH * scale) / 2;
  return (nx, ny) => [nx * videoW * scale + offX, ny * videoH * scale + offY];
}

// Returns null when the foot is not confidently in view.
export function footAnchor(landmarks, worldLandmarks, side, toScreen, minVisibility = 0.5) {
  const ids = FOOT_LANDMARKS[side];
  const pts = [ids.heel, ids.toe, ids.ankle].map((i) => landmarks[i]);
  if (pts.some((p) => !p || (p.visibility ?? 1) < minVisibility)) return null;
  const [heel2d, toe2d] = pts.map((p) => toScreen(p.x, p.y));

  const heelW = toThree(worldLandmarks[ids.heel]);
  const toeW = toThree(worldLandmarks[ids.toe]);
  const ankleW = toThree(worldLandmarks[ids.ankle]);

  // Shoe frame: +Z toward the toe, +Y up the leg, +X to the side.
  const forward = norm(sub(toeW, heelW));
  let up = sub(ankleW, heelW);
  up = norm(sub(up, forward.map((f) => f * dot(up, forward))));
  const right = cross(up, forward);

  // Share of the foot's length that lies in the screen plane; used to turn
  // on-screen length back into true length when the foot is foreshortened.
  const inPlane = Math.max(Math.hypot(forward[0], forward[1]), 0.3);
  const screenLength = Math.hypot(toe2d[0] - heel2d[0], toe2d[1] - heel2d[1]);

  return {
    heel: heel2d,
    toe: toe2d,
    basis: { right, up, forward },
    pixelsPerFootLength: screenLength / inPlane,
    values() {
      return [...heel2d, ...right, ...up, ...forward, this.pixelsPerFootLength];
    },
  };
}
