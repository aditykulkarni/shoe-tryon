import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sizesFromLength, widthClass } from '../src/lib/sizing.js';

test('265 mm foot lands on common sizes', () => {
  const s = sizesFromLength(265);
  assert.equal(s.mondopoint, 265);
  assert.equal(s.eu, 42);
  assert.equal(s.usMen, 9.5);
  assert.equal(s.usWomen, 11);
  assert.equal(s.uk, 9);
});

test('width classes by ratio', () => {
  assert.equal(widthClass(265, 90), 'narrow');
  assert.equal(widthClass(265, 100), 'standard');
  assert.equal(widthClass(265, 110), 'wide');
});
