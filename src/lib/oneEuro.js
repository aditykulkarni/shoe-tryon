// One Euro filter (Casiez et al. 2012): low lag when moving fast, low
// jitter when still. One instance per scalar signal.
const alpha = (cutoff, dt) => {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
};

export class OneEuroFilter {
  constructor({ minCutoff = 1.0, beta = 0.02, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }

  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }

  filter(value, tSeconds) {
    if (this.x === null) {
      this.x = value;
      this.t = tSeconds;
      return value;
    }
    const dt = Math.max(tSeconds - this.t, 1e-3);
    this.t = tSeconds;
    const dxRaw = (value - this.x) / dt;
    const ad = alpha(this.dCutoff, dt);
    this.dx = ad * dxRaw + (1 - ad) * this.dx;
    const a = alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt);
    this.x = a * value + (1 - a) * this.x;
    return this.x;
  }
}

// Filters an array of numbers element-wise.
export class VectorFilter {
  constructor(size, opts) {
    this.filters = Array.from({ length: size }, () => new OneEuroFilter(opts));
  }

  reset() {
    this.filters.forEach((f) => f.reset());
  }

  filter(values, t) {
    return values.map((v, i) => this.filters[i].filter(v, t));
  }
}
