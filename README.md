# AR shoe try-on (web prototype)

Mobile web app with two screens:

- **Measure**: photograph your foot standing on a sheet of A4 or US Letter paper, tap the paper's corners and a few foot points, and get foot length, width and rule-of-thumb EU/US/UK sizes. The result is saved on the phone.
- **Try on**: the rear camera runs MediaPipe Pose on-device, takes the heel, toe and ankle points of each foot, smooths them, and draws the shoe over the video with three.js, scaled to the measured foot.

## Run it on your phone

```sh
npm install
npm run dev
```

Vite prints a `Network:` URL like `https://192.168.1.20:5173`. Open it on a phone on the same Wi-Fi and accept the self-signed certificate warning; camera access needs HTTPS. The first try-on load downloads the pose model (about 9 MB) from Google's model storage.

`npm run build` writes a static site to `dist/` that can be hosted anywhere with HTTPS. Every push to `main` runs the tests and deploys to GitHub Pages through `.github/workflows/pages.yml` (set **Settings → Pages → Source** to **GitHub Actions** once).

## Use your own shoe

Export from 3DEXPERIENCE, clean it up, and save it as `public/models/shoe.glb`. Conventions are in [public/models/README.md](public/models/README.md).

## Layout

| Path | What it does |
| --- | --- |
| `src/measure.js` | Measure screen: photo, taps, results |
| `src/tryon.js` | Camera, pose tracking, three.js render loop |
| `src/shoe.js` | Loads `shoe.glb` or builds the placeholder |
| `src/lib/homography.js` | Paper-plane homography and mm distances |
| `src/lib/sizing.js` | Paper sizes, size conversions, width class |
| `src/lib/footPose.js` | Landmarks to per-foot position, orientation and scale |
| `src/lib/oneEuro.js` | Jitter filter for the tracked pose |
| `test/` | `npm test` (Node's built-in test runner) |

## Known limits

- MediaPipe Pose is a full-body model. It tracks feet best when the lower legs are in frame (a friend holding the phone, or the phone propped on the floor facing you), and loses them when you look straight down at your own feet. A dedicated foot model is the planned fix.
- No occlusion yet: the shoe draws over the ankle and trousers.
- Size conversions are generic formulas; replace them with your brand's size chart.
