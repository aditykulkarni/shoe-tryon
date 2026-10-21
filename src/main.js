import './style.css';
import { startMeasure } from './measure.js';

const view = document.querySelector('#view');
const tabs = document.querySelectorAll('nav a');
let stop = () => {};

async function route() {
  stop();
  const page = location.hash === '#tryon' ? 'tryon' : 'measure';
  tabs.forEach((a) => a.classList.toggle('active', a.hash === `#${page}`));
  document.body.dataset.page = page;
  if (page === 'tryon') {
    // Loaded on demand: three.js and MediaPipe are most of the bundle.
    const { startTryOn } = await import('./tryon.js');
    stop = await startTryOn(view);
  } else {
    stop = startMeasure(view, { onDone: () => (location.hash = '#tryon') });
  }
}

window.addEventListener('hashchange', route);
route();
