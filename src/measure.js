import { measureMm, paperHomography } from './lib/homography.js';
import { PAPERS, sizesFromLength, widthClass } from './lib/sizing.js';
import { saveMeasurement } from './lib/storage.js';

const STEPS = [
  { key: 'corner', count: 4, prompt: 'Tap the four corners of the paper, in any order.' },
  { key: 'heel', count: 1, prompt: 'Tap the very back of your heel, where it meets the paper.' },
  { key: 'toe', count: 1, prompt: 'Tap the tip of your longest toe.' },
  {
    key: 'width',
    count: 2,
    prompt: 'Tap the inside edge of your foot at the big-toe joint, then the outside edge at the little-toe joint (the widest part).',
  },
];
const COLORS = { corner: '#f6ad55', heel: '#68d391', toe: '#68d391', width: '#63b3ed' };

export function startMeasure(root, { onDone }) {
  root.innerHTML = `
    <div class="measure">
      <p class="lead">Put a sheet of paper flat on the floor. Stand on it barefoot or in thin socks, heel a couple of centimetres in from the edge, so paper shows all around your foot. Have someone take a photo from directly above (or lean over and hold the phone level over your foot). All four corners of the paper must be in the photo.</p>
      <div class="row">
        <label>Paper
          <select name="paper">
            ${Object.entries(PAPERS).map(([k, p]) => `<option value="${k}">${p.label}</option>`).join('')}
          </select>
        </label>
        <label class="button">Take photo
          <input type="file" accept="image/*" capture="environment" hidden>
        </label>
      </div>
      <p class="prompt" hidden></p>
      <canvas hidden></canvas>
      <div class="row actions" hidden>
        <button type="button" data-act="undo">Undo tap</button>
        <button type="button" data-act="restart">Start over</button>
      </div>
      <div class="result" hidden></div>
    </div>`;

  const canvas = root.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const prompt = root.querySelector('.prompt');
  const actions = root.querySelector('.actions');
  const result = root.querySelector('.result');
  const paperSelect = root.querySelector('select');
  let image = null;
  let taps = [];

  const stepFor = (n) => {
    let seen = 0;
    for (const step of STEPS) {
      if (n < seen + step.count) return step;
      seen += step.count;
    }
    return null;
  };
  const totalTaps = STEPS.reduce((s, st) => s + st.count, 0);

  const draw = () => {
    ctx.drawImage(image, 0, 0);
    const r = Math.max(canvas.width, canvas.height) / 120;
    taps.forEach((p, i) => {
      ctx.fillStyle = COLORS[stepFor(i).key];
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    });
    const next = stepFor(taps.length);
    prompt.textContent = next ? next.prompt : 'Done. Your measurement is below.';
  };

  const finish = () => {
    const paper = PAPERS[paperSelect.value];
    const H = paperHomography(taps.slice(0, 4), paper);
    const lengthMm = Math.round(measureMm(H, taps[4], taps[5]));
    const widthMm = Math.round(measureMm(H, taps[6], taps[7]));
    const sizes = sizesFromLength(lengthMm);
    const measurement = { lengthMm, widthMm, paper: paperSelect.value, at: new Date().toISOString() };
    saveMeasurement(measurement);
    result.hidden = false;
    result.innerHTML = `
      <dl>
        <dt>Length</dt><dd>${lengthMm} mm</dd>
        <dt>Width</dt><dd>${widthMm} mm (${widthClass(lengthMm, widthMm)})</dd>
        <dt>EU</dt><dd>${sizes.eu}</dd>
        <dt>US men / women</dt><dd>${sizes.usMen} / ${sizes.usWomen}</dd>
        <dt>UK</dt><dd>${sizes.uk}</dd>
      </dl>
      <p class="note">Sizes use rule-of-thumb conversions, not a brand size chart.</p>
      <button type="button" data-act="tryon">Try on the shoe</button>`;
    result.querySelector('[data-act="tryon"]').onclick = () => onDone(measurement);
  };

  const reset = () => {
    taps = [];
    result.hidden = true;
    if (image) draw();
  };

  root.querySelector('input[type="file"]').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    image = await createImageBitmap(file, { imageOrientation: 'from-image' });
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.hidden = false;
    prompt.hidden = false;
    actions.hidden = false;
    reset();
  };

  canvas.onclick = (e) => {
    if (!image || taps.length >= totalTaps) return;
    const rect = canvas.getBoundingClientRect();
    taps.push([
      ((e.clientX - rect.left) / rect.width) * canvas.width,
      ((e.clientY - rect.top) / rect.height) * canvas.height,
    ]);
    draw();
    if (taps.length === totalTaps) {
      try {
        finish();
      } catch (err) {
        prompt.textContent = `${err.message}. Start over and tap the corners again.`;
      }
    }
  };

  actions.onclick = (e) => {
    const act = e.target.dataset.act;
    if (act === 'undo' && taps.length) {
      taps.pop();
      result.hidden = true;
      draw();
    } else if (act === 'restart') {
      reset();
    }
  };
  paperSelect.onchange = () => {
    if (taps.length === totalTaps) finish();
  };

  return () => {};
}
