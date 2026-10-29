import test from 'node:test';
import assert from 'node:assert/strict';
import {
  subbandToSpatialBlock,
  computeHaarHH,
  validateImageDimensions,
  computePixelDifference,
  isValidSubbandCoord,
  calculateImageViewport,
  mapSubbandToCanvas,
} from './differenceEngine.ts';

test('subbandToSpatialBlock correctly maps 256x256 subband to 2x2 spatial block', () => {
  // Top-left corner (0, 0)
  const block0 = subbandToSpatialBlock(0, 0);
  assert.deepEqual(block0, { minX: 0, maxX: 1, minY: 0, maxY: 1 });

  // Arbitrary coordinate (10, 25)
  const blockArbitrary = subbandToSpatialBlock(10, 25);
  assert.deepEqual(blockArbitrary, { minX: 50, maxX: 51, minY: 20, maxY: 21 });

  // Bottom-right corner (255, 255) in 512x512 image
  const blockLast = subbandToSpatialBlock(255, 255);
  assert.deepEqual(blockLast, { minX: 510, maxX: 511, minY: 510, maxY: 511 });
});

test('computeHaarHH accurately calculates (a - b - c + d) / 2.0', () => {
  // Create a 2x2 RGBA image buffer (width=2, height=2, 16 bytes)
  const width = 2;
  const pixels = new Uint8ClampedArray(16);

  // Set Red channel values:
  // a: [0, 0] = 100
  // b: [0, 1] = 60
  // c: [1, 0] = 40
  // d: [1, 1] = 20
  pixels[0] = 100; // a (y=0, x=0)
  pixels[4] = 60;  // b (y=0, x=1)
  pixels[8] = 40;  // c (y=1, x=0)
  pixels[12] = 20; // d (y=1, x=1)

  // HH = (100 - 60 - 40 + 20) / 2.0 = 20 / 2.0 = 10.0
  const hhVal = computeHaarHH(pixels, width, 0, 0, 0);
  assert.equal(hhVal, 10.0);

  // Check when values are equal (flat patch): HH = 0
  pixels[0] = 50;
  pixels[4] = 50;
  pixels[8] = 50;
  pixels[12] = 50;
  assert.equal(computeHaarHH(pixels, width, 0, 0, 0), 0.0);
});

test('validateImageDimensions accepts matching dimensions and rejects mismatches', () => {
  // Compatible dimensions
  assert.doesNotThrow(() => validateImageDimensions(512, 512, 512, 512));
  assert.doesNotThrow(() => validateImageDimensions(256, 256, 256, 256));

  // Incompatible dimensions
  assert.throws(
    () => validateImageDimensions(512, 512, 256, 256),
    /Dimension mismatch: Carrier is 512x512, but Stego is 256x256/
  );
  assert.throws(
    () => validateImageDimensions(512, 512, 512, 500),
    /Dimension mismatch/
  );

  // Non-positive dimensions
  assert.throws(() => validateImageDimensions(0, 512, 0, 512), /Invalid image dimensions/);
});

test('computePixelDifference reports 0 changed pixels for identical images', () => {
  const width = 4;
  const height = 4;
  const carrier = new Uint8ClampedArray(width * height * 4).fill(128);
  const stego = new Uint8ClampedArray(width * height * 4).fill(128);

  const { metrics, outputPixels } = computePixelDifference(carrier, stego, width, height, { mode: 'grayscale' });

  assert.equal(metrics.totalPixels, 16);
  assert.equal(metrics.changedPixelsCount, 0);
  assert.equal(metrics.maxChannelDiff, 0);
  assert.equal(metrics.changedPercentage, 0);

  // Output pixels should all have 0 diff in RGB
  for (let i = 0; i < outputPixels.length; i += 4) {
    assert.equal(outputPixels[i], 0);
    assert.equal(outputPixels[i + 1], 0);
    assert.equal(outputPixels[i + 2], 0);
  }
});

test('computePixelDifference detects modified pixels and applies amplification', () => {
  const width = 2;
  const height = 2;
  const carrier = new Uint8ClampedArray(width * height * 4);
  const stego = new Uint8ClampedArray(width * height * 4);

  // Modify pixel 0: Red diff = 2
  stego[0] = 2; // R
  // Modify pixel 3: Green diff = 5, Blue diff = 1
  stego[13] = 5; // G (pixel index 3 is byte offset 12: R=12, G=13, B=14)
  stego[14] = 1; // B

  const { metrics, outputPixels } = computePixelDifference(carrier, stego, width, height, {
    mode: 'amplified',
    amplificationFactor: 10,
  });

  assert.equal(metrics.totalPixels, 4);
  assert.equal(metrics.changedPixelsCount, 2);
  assert.equal(metrics.maxChannelDiff, 5);
  assert.equal(metrics.changedPercentage, 50);

  // Pixel 0 should be amplified: R was 2, amp=10 -> 20
  assert.equal(outputPixels[0], 20);

  // Pixel 3 should be amplified: G was 5, amp=10 -> 50
  assert.equal(outputPixels[13], 50);
});

test('computePixelDifference rejects channel mismatch and buffer length mismatch', () => {
  const w = 4;
  const h = 4;
  const carrierRgba = new Uint8ClampedArray(w * h * 4);
  const stegoRgb = new Uint8ClampedArray(w * h * 3);

  // Channel mismatch
  assert.throws(
    () => computePixelDifference(carrierRgba, stegoRgb, { width: w, height: h, channels: 4 }, { width: w, height: h, channels: 3 }),
    /Channel mismatch/
  );

  // Buffer length mismatch for carrier
  const truncatedCarrier = new Uint8ClampedArray(w * h * 4 - 4);
  assert.throws(
    () => computePixelDifference(truncatedCarrier, carrierRgba, { width: w, height: h, channels: 4 }),
    /Carrier buffer length/
  );

  // Buffer length mismatch for stego
  const truncatedStego = new Uint8ClampedArray(w * h * 4 - 4);
  assert.throws(
    () => computePixelDifference(carrierRgba, truncatedStego, { width: w, height: h, channels: 4 }),
    /Stego buffer length/
  );
});

test('computePixelDifference supports 3-channel RGB inputs and outputs RGBA', () => {
  const w = 2;
  const h = 2;
  // 4 pixels * 3 channels = 12 bytes
  const carrierRgb = new Uint8Array([
    10, 20, 30,
    40, 50, 60,
    70, 80, 90,
    100, 110, 120,
  ]);
  const stegoRgb = new Uint8Array([
    10, 20, 32, // Blue diff = 2
    40, 55, 60, // Green diff = 5
    70, 80, 90, // Identical
    100, 110, 120, // Identical
  ]);

  const { metrics, outputPixels } = computePixelDifference(
    carrierRgb,
    stegoRgb,
    { width: w, height: h, channels: 3 },
    { width: w, height: h, channels: 3 },
    { mode: 'amplified', amplificationFactor: 10 }
  );

  assert.equal(metrics.totalPixels, 4);
  assert.equal(metrics.changedPixelsCount, 2);
  assert.equal(metrics.maxChannelDiff, 5);
  assert.deepEqual(metrics.carrierDimensions, { width: 2, height: 2 });
  assert.deepEqual(metrics.stegoDimensions, { width: 2, height: 2 });

  // Output must always be 4-channel RGBA (16 bytes)
  assert.equal(outputPixels.length, 16);
  // Alpha channel: 255 for changed pixels, 40 for unchanged background
  assert.equal(outputPixels[3], 255);
  assert.equal(outputPixels[7], 255);
  assert.equal(outputPixels[11], 40);
  assert.equal(outputPixels[15], 40);
});

test('isValidSubbandCoord validates coordinates strictly within bounds and rejects invalid inputs', () => {
  const subH = 678;
  const subW = 1020;

  // Valid corner coordinates
  assert.equal(isValidSubbandCoord(0, 0, subH, subW), true);
  assert.equal(isValidSubbandCoord(0, subW - 1, subH, subW), true);
  assert.equal(isValidSubbandCoord(subH - 1, 0, subH, subW), true);
  assert.equal(isValidSubbandCoord(subH - 1, subW - 1, subH, subW), true);
  assert.equal(isValidSubbandCoord(100, 200, subH, subW), true);

  // Negative coordinates -> rejected
  assert.equal(isValidSubbandCoord(-1, 0, subH, subW), false);
  assert.equal(isValidSubbandCoord(0, -1, subH, subW), false);
  assert.equal(isValidSubbandCoord(-5, -5, subH, subW), false);

  // Out of bounds (>= dimension) -> rejected
  assert.equal(isValidSubbandCoord(subH, 0, subH, subW), false);
  assert.equal(isValidSubbandCoord(0, subW, subH, subW), false);
  assert.equal(isValidSubbandCoord(subH + 10, subW + 10, subH, subW), false);

  // NaN or non-number -> rejected
  assert.equal(isValidSubbandCoord(NaN, 0, subH, subW), false);
  assert.equal(isValidSubbandCoord(0, NaN, subH, subW), false);
  assert.equal(isValidSubbandCoord(undefined as any, 0, subH, subW), false);
});

test('calculateImageViewport accurately computes letterboxing, pillarboxing, and 1:1 viewports', () => {

  // 1. Exact 1:1 match (512x512 container, 512x512 image)
  const vp1 = calculateImageViewport(512, 512, 512, 512, 'contain');
  assert.equal(vp1.offsetX, 0);
  assert.equal(vp1.offsetY, 0);
  assert.equal(vp1.displayedWidth, 512);
  assert.equal(vp1.displayedHeight, 512);
  assert.equal(vp1.scaleX, 1);
  assert.equal(vp1.scaleY, 1);

  // 2. Letterbox: 800x800 square container with 2040x1356 landscape image
  // Aspect ratio = 2040 / 1356 = 1.5044... Scale limited by width (800 / 2040 = 0.39215686)
  const vpLandscape = calculateImageViewport(800, 800, 2040, 1356, 'contain');
  assert.equal(vpLandscape.offsetX, 0);
  assert.equal(vpLandscape.displayedWidth, 800);
  assert.equal(Math.round(vpLandscape.displayedHeight), Math.round(1356 * (800 / 2040))); // ~532px
  assert.ok(vpLandscape.offsetY > 130 && vpLandscape.offsetY < 135); // centered vertically ~134px margin top and bottom
  assert.equal(vpLandscape.offsetY + vpLandscape.displayedHeight + vpLandscape.offsetY, 800);

  // 3. Pillarbox: 800x800 square container with 512x768 portrait image
  // Scale limited by height (800 / 768 = 1.041666...)
  const vpPortrait = calculateImageViewport(800, 800, 512, 768, 'contain');
  assert.equal(vpPortrait.offsetY, 0);
  assert.equal(vpPortrait.displayedHeight, 800);
  assert.ok(vpPortrait.offsetX > 130 && vpPortrait.offsetX < 135); // centered horizontally
  assert.equal(Math.round(vpPortrait.offsetX + vpPortrait.displayedWidth + vpPortrait.offsetX), 800);
});

test('mapSubbandToCanvas maps corners strictly inside carrier dimensions and rejects invalid coordinates', () => {

  // Test 1: Non-square carrier 2040x1356 (subband 1020x678) in 1:1 viewport
  const vp1 = calculateImageViewport(2040, 1356, 2040, 1356, 'contain');
  const subH = 678;
  const subW = 1020;

  // Corner (0, 0) -> top-left
  const mTopLeft = mapSubbandToCanvas(0, 0, subH, subW, vp1);
  assert.ok(mTopLeft);
  assert.equal(mTopLeft.x, 0);
  assert.equal(mTopLeft.y, 0);
  assert.equal(mTopLeft.width, 2);
  assert.equal(mTopLeft.height, 2);

  // Corner (0, max_c): top-right (0, 1019)
  const mTopRight = mapSubbandToCanvas(0, subW - 1, subH, subW, vp1);
  assert.ok(mTopRight);
  assert.equal(mTopRight.x, 2 * 1019); // 2038
  assert.equal(mTopRight.y, 0);
  assert.equal(mTopRight.x + mTopRight.width, 2040); // strictly touches right boundary

  // Corner (max_r, 0): bottom-left (677, 0)
  const mBottomLeft = mapSubbandToCanvas(subH - 1, 0, subH, subW, vp1);
  assert.ok(mBottomLeft);
  assert.equal(mBottomLeft.x, 0);
  assert.equal(mBottomLeft.y, 2 * 677); // 1354
  assert.equal(mBottomLeft.y + mBottomLeft.height, 1356); // strictly touches bottom boundary

  // Corner (max_r, max_c): bottom-right (677, 1019)
  const mBottomRight = mapSubbandToCanvas(subH - 1, subW - 1, subH, subW, vp1);
  assert.ok(mBottomRight);
  assert.equal(mBottomRight.x + mBottomRight.width, 2040);
  assert.equal(mBottomRight.y + mBottomRight.height, 1356);

  // Test 2: Out-of-bounds coordinates must return null (strict rejection, NO clamping)
  assert.equal(mapSubbandToCanvas(-1, 0, subH, subW, vp1), null);
  assert.equal(mapSubbandToCanvas(0, -1, subH, subW, vp1), null);
  assert.equal(mapSubbandToCanvas(subH, 0, subH, subW, vp1), null);
  assert.equal(mapSubbandToCanvas(0, subW, subH, subW, vp1), null);
  assert.equal(mapSubbandToCanvas(subH + 5, subW + 5, subH, subW, vp1), null);

  // Test 3: Letterboxed container (800x800 container with 2040x1356 image)
  // Markers must strictly stay inside the displayed image rectangle [offsetX, offsetY, displayedW, displayedH]
  const vpLetterbox = calculateImageViewport(800, 800, 2040, 1356, 'contain');
  const mLetterboxTop = mapSubbandToCanvas(0, 0, subH, subW, vpLetterbox);
  assert.ok(mLetterboxTop);
  assert.ok(mLetterboxTop.y >= vpLetterbox.offsetY); // Never above displayed image!

  const mLetterboxBottom = mapSubbandToCanvas(subH - 1, subW - 1, subH, subW, vpLetterbox);
  assert.ok(mLetterboxBottom);
  assert.ok(mLetterboxBottom.y + mLetterboxBottom.height <= vpLetterbox.offsetY + vpLetterbox.displayedHeight + 0.001); // Never below displayed image!
});
