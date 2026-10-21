import * as THREE from 'three';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { coverMapper, footAnchor } from './lib/footPose.js';
import { VectorFilter } from './lib/oneEuro.js';
import { loadMeasurement } from './lib/storage.js';
import { loadShoe } from './shoe.js';

// Copied from node_modules by scripts/copy-wasm.js.
const WASM_URL = `${import.meta.env.BASE_URL}mediapipe`;
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const DEFAULT_FOOT_MM = 265;
// Extra length of the shoe over the foot, matching the sizing rule.
const SHOE_ALLOWANCE_MM = 15;

export async function startTryOn(root) {
  root.innerHTML = `
    <div class="tryon">
      <video playsinline muted></video>
      <canvas></canvas>
      <div class="hud"><span class="status">Starting camera…</span></div>
    </div>`;
  const video = root.querySelector('video');
  const canvas = root.querySelector('canvas');
  const status = root.querySelector('.status');
  const say = (text) => (status.textContent = text);

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    say(`Camera unavailable: ${err.message}. The page must be opened over HTTPS.`);
    return () => {};
  }
  video.srcObject = stream;
  await video.play();

  say('Loading foot tracking…');
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
  });

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(0.3, 1, 0.8);
  scene.add(sun);

  // Orthographic camera in screen pixels, y up, so landmarks map directly.
  const camera = new THREE.OrthographicCamera(0, 1, 0, -1, -5000, 5000);
  const resize = () => {
    const w = root.clientWidth;
    const h = root.clientHeight;
    renderer.setSize(w, h, false);
    camera.right = w;
    camera.bottom = -h;
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener('resize', resize);

  const shoe = await loadShoe();
  const measured = loadMeasurement();
  const footMm = measured?.lengthMm ?? DEFAULT_FOOT_MM;
  // Pixels per foot length times shoe length per foot length, per model unit.
  const shoeScale = (footMm + SHOE_ALLOWANCE_MM) / footMm / shoe.length;

  const feet = ['left', 'right'].map((side) => {
    const object = shoe.object.clone(true);
    object.visible = false;
    object.matrixAutoUpdate = false;
    scene.add(object);
    // The model is a right shoe; mirror it for the left foot.
    return { side, object, filter: new VectorFilter(12, { minCutoff: 1.2, beta: 0.01 }), mirror: side === 'left' ? -1 : 1 };
  });

  say(
    `${shoe.isPlaceholder ? 'Placeholder shoe' : 'Your shoe'} · foot ${footMm} mm${measured ? '' : ' (default, measure first)'} · point the camera at your feet`
  );

  const basis = new THREE.Matrix4();
  const scale = new THREE.Matrix4();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const forward = new THREE.Vector3();

  let running = true;
  let lastVideoTime = -1;
  const loop = () => {
    if (!running) return;
    requestAnimationFrame(loop);
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    const landmarks = result.landmarks[0];
    const world = result.worldLandmarks[0];
    const toScreen = coverMapper(video.videoWidth, video.videoHeight, root.clientWidth, root.clientHeight);

    for (const foot of feet) {
      const anchor = landmarks && world ? footAnchor(landmarks, world, foot.side, toScreen) : null;
      if (!anchor) {
        foot.object.visible = false;
        foot.filter.reset();
        continue;
      }
      const v = foot.filter.filter(anchor.values(), now / 1000);
      forward.set(v[8], v[9], v[10]).normalize();
      up.set(v[5], v[6], v[7]);
      up.addScaledVector(forward, -up.dot(forward)).normalize();
      right.crossVectors(up, forward);

      const s = v[11] * shoeScale;
      basis.makeBasis(right, up, forward).setPosition(v[0], -v[1], 0);
      scale.makeScale(s * foot.mirror, s, s);
      foot.object.matrix.multiplyMatrices(basis, scale);
      foot.object.visible = true;
    }
    renderer.render(scene, camera);
  };
  loop();

  return () => {
    running = false;
    window.removeEventListener('resize', resize);
    stream.getTracks().forEach((t) => t.stop());
    landmarker.close();
    renderer.dispose();
  };
}
