// Copies MediaPipe's wasm runtime into public/ so it is served with the app
// and always matches the installed JS package.
import { cpSync } from 'node:fs';

cpSync('node_modules/@mediapipe/tasks-vision/wasm', 'public/mediapipe', { recursive: true });
