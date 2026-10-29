/**
 * Difference Engine and Transform Mapping Utilities for PaladinMedia Steganography Inspection.
 *
 * Provides:
 * 1. Pixel-level difference calculation between carrier and stego images.
 * 2. Visual difference map generation (grayscale, amplified, heatmap).
 * 3. 1-level 2D Haar DWT spatial-to-frequency coordinate mapping:
 *    (row, col) in 256x256 subband <-> 2x2 spatial block [2r..2r+1, 2c..2c+1] in 512x512 carrier.
 * 4. Single-coefficient Haar DWT computation: HH = (a - b - c + d) / 2.0.
 */

export interface DiffMetrics {
  totalPixels: number;
  changedPixelsCount: number;
  maxChannelDiff: number;
  changedPercentage: number;
  carrierDimensions: { width: number; height: number };
  stegoDimensions: { width: number; height: number };
}

export interface DiffOptions {
  mode: "grayscale" | "amplified" | "heatmap";
  amplificationFactor?: number;
}

export interface ImageMetadata {
  width: number;
  height: number;
  channels?: number; // 3 (RGB) or 4 (RGBA), default 4
}

export interface SubbandCoord {
  r: number;
  c: number;
}

export interface SpatialBlock {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Maps a 256x256 subband coordinate (r, c) to its corresponding 2x2 spatial block
 * in the 512x512 carrier image according to 1-level 2D Haar DWT support.
 */
export function subbandToSpatialBlock(r: number, c: number): SpatialBlock {
  return {
    minX: 2 * c,
    maxX: 2 * c + 1,
    minY: 2 * r,
    maxY: 2 * r + 1,
  };
}

/**
 * Validates whether a subband coordinate (r, c) is strictly within valid subband bounds.
 */
export function isValidSubbandCoord(
  r: number,
  c: number,
  subbandHeight: number,
  subbandWidth: number
): boolean {
  if (typeof r !== "number" || typeof c !== "number" || isNaN(r) || isNaN(c)) return false;
  return r >= 0 && r < subbandHeight && c >= 0 && c < subbandWidth;
}

/**
 * Validates whether a spatial block is strictly within carrier image bounds.
 */
export function isSpatialBlockWithinBounds(
  block: SpatialBlock,
  carrierWidth: number,
  carrierHeight: number
): boolean {
  return (
    block.minX >= 0 &&
    block.maxX < carrierWidth &&
    block.minY >= 0 &&
    block.maxY < carrierHeight &&
    block.minX <= block.maxX &&
    block.minY <= block.maxY
  );
}

/**
 * Represents the calculated display rectangle and scaling factors for an image
 * fitted into a display container (e.g. via object-fit contain).
 */
export interface ImageViewport {
  offsetX: number;
  offsetY: number;
  displayedWidth: number;
  displayedHeight: number;
  naturalWidth: number;
  naturalHeight: number;
  scaleX: number;
  scaleY: number;
}

/**
 * Calculates the exact displayed viewport rectangle and scaling factors
 * for an image placed inside a container box with optional aspect-ratio containment.
 */
export function calculateImageViewport(
  containerWidth: number,
  containerHeight: number,
  naturalWidth: number,
  naturalHeight: number,
  fit: "contain" | "fill" | "none" = "contain"
): ImageViewport {
  if (naturalWidth <= 0 || naturalHeight <= 0 || containerWidth <= 0 || containerHeight <= 0) {
    return {
      offsetX: 0,
      offsetY: 0,
      displayedWidth: Math.max(0, containerWidth),
      displayedHeight: Math.max(0, containerHeight),
      naturalWidth: Math.max(1, naturalWidth),
      naturalHeight: Math.max(1, naturalHeight),
      scaleX: 1,
      scaleY: 1,
    };
  }

  if (fit === "none") {
    return {
      offsetX: 0,
      offsetY: 0,
      displayedWidth: naturalWidth,
      displayedHeight: naturalHeight,
      naturalWidth,
      naturalHeight,
      scaleX: 1,
      scaleY: 1,
    };
  }

  if (fit === "fill") {
    return {
      offsetX: 0,
      offsetY: 0,
      displayedWidth: containerWidth,
      displayedHeight: containerHeight,
      naturalWidth,
      naturalHeight,
      scaleX: containerWidth / naturalWidth,
      scaleY: containerHeight / naturalHeight,
    };
  }

  // "contain": scale uniformly to fit within container, centered with letterboxing/pillarboxing
  const scale = Math.min(containerWidth / naturalWidth, containerHeight / naturalHeight);
  const displayedWidth = naturalWidth * scale;
  const displayedHeight = naturalHeight * scale;
  const offsetX = (containerWidth - displayedWidth) / 2;
  const offsetY = (containerHeight - displayedHeight) / 2;

  return {
    offsetX,
    offsetY,
    displayedWidth,
    displayedHeight,
    naturalWidth,
    naturalHeight,
    scaleX: scale,
    scaleY: scale,
  };
}

/**
 * Maps a subband coordinate (r, c) to canvas coordinates according to an explicit ImageViewport.
 * Returns null if the coordinate is out of bounds (strict rejection, NO clamping of invalid coordinates).
 */
export function mapSubbandToCanvas(
  r: number,
  c: number,
  subbandHeight: number,
  subbandWidth: number,
  viewport: ImageViewport
): { x: number; y: number; width: number; height: number } | null {
  if (!isValidSubbandCoord(r, c, subbandHeight, subbandWidth)) {
    return null;
  }

  const block = subbandToSpatialBlock(r, c);
  if (!isSpatialBlockWithinBounds(block, viewport.naturalWidth, viewport.naturalHeight)) {
    return null;
  }

  const rawX = viewport.offsetX + block.minX * viewport.scaleX;
  const rawY = viewport.offsetY + block.minY * viewport.scaleY;
  // Each subband coefficient corresponds to a 2x2 spatial block
  const rawWidth = Math.max(1, 2 * viewport.scaleX);
  const rawHeight = Math.max(1, 2 * viewport.scaleY);

  const minBoundX = viewport.offsetX;
  const maxBoundX = viewport.offsetX + viewport.displayedWidth;
  const minBoundY = viewport.offsetY;
  const maxBoundY = viewport.offsetY + viewport.displayedHeight;

  // Ensure marker strictly stays within the displayed viewport bounds
  const x = Math.max(minBoundX, rawX);
  const y = Math.max(minBoundY, rawY);
  const width = Math.min(rawWidth, maxBoundX - x);
  const height = Math.min(rawHeight, maxBoundY - y);

  if (width <= 0 || height <= 0 || x >= maxBoundX || y >= maxBoundY) {
    return null;
  }

  return { x, y, width, height };
}

/**
 * Computes the 1-level 2D Haar HH coefficient for a 2x2 block on a single channel.
 * Formulation: HH = (a - b - c + d) / 2.0
 * where:
 *   a = [2r, 2c]       (top-left)
 *   b = [2r, 2c + 1]   (top-right)
 *   c = [2r + 1, 2c]   (bottom-left)
 *   d = [2r + 1, 2c + 1] (bottom-right)
 */
export function computeHaarHH(
  pixels: Uint8ClampedArray | Uint8Array,
  imageWidth: number,
  r: number,
  c: number,
  channel: number = 0 // 0 = Red channel
): number {
  const stride = imageWidth * 4;
  const idxA = (2 * r) * stride + (2 * c) * 4 + channel;
  const idxB = (2 * r) * stride + (2 * c + 1) * 4 + channel;
  const idxC = (2 * r + 1) * stride + (2 * c) * 4 + channel;
  const idxD = (2 * r + 1) * stride + (2 * c + 1) * 4 + channel;

  const a = pixels[idxA];
  const b = pixels[idxB];
  const cVal = pixels[idxC];
  const d = pixels[idxD];

  return (a - b - cVal + d) / 2.0;
}

/**
 * Validates dimensions of carrier and stego image data.
 * Throws an Error with a descriptive explanation if dimensions are incompatible.
 */
export function validateImageDimensions(
  carrierWidth: number,
  carrierHeight: number,
  stegoWidth: number,
  stegoHeight: number,
  expectedWidth?: number,
  expectedHeight?: number
): void {
  if (carrierWidth <= 0 || carrierHeight <= 0 || stegoWidth <= 0 || stegoHeight <= 0) {
    throw new Error("Invalid image dimensions: dimensions must be greater than zero.");
  }
  if (carrierWidth !== stegoWidth || carrierHeight !== stegoHeight) {
    throw new Error(
      `Dimension mismatch: Carrier is ${carrierWidth}x${carrierHeight}, but Stego is ${stegoWidth}x${stegoHeight}. Exact pixel comparison requires matching dimensions.`
    );
  }
  if (expectedWidth !== undefined && (carrierWidth !== expectedWidth || stegoWidth !== expectedWidth)) {
    throw new Error(
      `Incompatible width: Expected ${expectedWidth}, but got Carrier: ${carrierWidth}, Stego: ${stegoWidth}.`
    );
  }
  if (expectedHeight !== undefined && (carrierHeight !== expectedHeight || stegoHeight !== expectedHeight)) {
    throw new Error(
      `Incompatible height: Expected ${expectedHeight}, but got Carrier: ${carrierHeight}, Stego: ${stegoHeight}.`
    );
  }
}

/**
 * Calculates per-pixel channel differences between carrier and stego buffers,
 * computes exact metrics, and renders the result into an output ImageData buffer.
 *
 * Supports input as (carrierData, stegoData, carrierMeta, stegoMeta, options)
 * or legacy (carrierData, stegoData, width, height, options).
 */
export function computePixelDifference(
  carrierData: Uint8ClampedArray | Uint8Array,
  stegoData: Uint8ClampedArray | Uint8Array,
  carrierMetaOrWidth: ImageMetadata | number,
  stegoMetaOrHeight?: ImageMetadata | number,
  options: DiffOptions = { mode: "grayscale", amplificationFactor: 10 }
): { outputPixels: Uint8ClampedArray; metrics: DiffMetrics } {
  let carrierW: number;
  let carrierH: number;
  let stegoW: number;
  let stegoH: number;
  let channels = 4;

  if (typeof carrierMetaOrWidth === "number") {
    carrierW = carrierMetaOrWidth;
    carrierH = typeof stegoMetaOrHeight === "number" ? stegoMetaOrHeight : carrierMetaOrWidth;
    stegoW = carrierW;
    stegoH = carrierH;
  } else {
    carrierW = carrierMetaOrWidth.width;
    carrierH = carrierMetaOrWidth.height;
    channels = carrierMetaOrWidth.channels || 4;
    if (typeof stegoMetaOrHeight === "object" && stegoMetaOrHeight !== null) {
      stegoW = stegoMetaOrHeight.width;
      stegoH = stegoMetaOrHeight.height;
      if (stegoMetaOrHeight.channels && stegoMetaOrHeight.channels !== channels) {
        throw new Error(
          `Channel mismatch: Carrier has ${channels} channels, but Stego has ${stegoMetaOrHeight.channels} channels.`
        );
      }
    } else {
      stegoW = carrierW;
      stegoH = carrierH;
    }
  }

  // Validate dimensions
  validateImageDimensions(carrierW, carrierH, stegoW, stegoH);

  if (channels !== 3 && channels !== 4) {
    throw new Error(`Unsupported channel layout: expected 3 (RGB) or 4 (RGBA), got ${channels}.`);
  }

  const totalPixels = carrierW * carrierH;
  const expectedLength = totalPixels * channels;
  if (carrierData.length !== expectedLength) {
    throw new Error(
      `Carrier buffer length (${carrierData.length}) does not match expected size (${expectedLength} bytes for ${carrierW}x${carrierH}x${channels}).`
    );
  }
  if (stegoData.length !== expectedLength) {
    throw new Error(
      `Stego buffer length (${stegoData.length}) does not match expected size (${expectedLength} bytes for ${stegoW}x${stegoH}x${channels}).`
    );
  }

  // Output is always 4-channel RGBA for HTML canvas ImageData compatibility
  const outputPixels = new Uint8ClampedArray(totalPixels * 4);
  const amp = options.amplificationFactor || 1;

  let changedCount = 0;
  let maxDiff = 0;

  for (let p = 0; p < totalPixels; p++) {
    const inIdx = p * channels;
    const outIdx = p * 4;

    const rDiff = Math.abs(stegoData[inIdx] - carrierData[inIdx]);
    const gDiff = Math.abs(stegoData[inIdx + 1] - carrierData[inIdx + 1]);
    const bDiff = Math.abs(stegoData[inIdx + 2] - carrierData[inIdx + 2]);

    const pixelMaxDiff = Math.max(rDiff, gDiff, bDiff);
    if (pixelMaxDiff > 0) {
      changedCount++;
      if (pixelMaxDiff > maxDiff) {
        maxDiff = pixelMaxDiff;
      }
    }

    if (options.mode === "grayscale") {
      const meanDiff = Math.round((rDiff + gDiff + bDiff) / 3);
      outputPixels[outIdx] = meanDiff;
      outputPixels[outIdx + 1] = meanDiff;
      outputPixels[outIdx + 2] = meanDiff;
      outputPixels[outIdx + 3] = 255;
    } else if (options.mode === "amplified") {
      const rAmp = Math.min(255, rDiff * amp);
      const gAmp = Math.min(255, gDiff * amp);
      const bAmp = Math.min(255, bDiff * amp);
      const maxAmp = Math.max(rAmp, gAmp, bAmp);
      outputPixels[outIdx] = rAmp > 0 ? rAmp : 0;
      outputPixels[outIdx + 1] = gAmp > 0 ? gAmp : 0;
      outputPixels[outIdx + 2] = bAmp > 0 ? bAmp : 0;
      outputPixels[outIdx + 3] = maxAmp > 0 ? 255 : 40;
    } else if (options.mode === "heatmap") {
      const normalized = Math.min(1.0, (pixelMaxDiff * amp) / 32.0);
      const [rRamp, gRamp, bRamp] = heatMapRamp(normalized);
      outputPixels[outIdx] = rRamp;
      outputPixels[outIdx + 1] = gRamp;
      outputPixels[outIdx + 2] = bRamp;
      outputPixels[outIdx + 3] = pixelMaxDiff > 0 ? 255 : 30;
    }
  }

  const changedPercentage = totalPixels > 0 ? Number(((changedCount / totalPixels) * 100).toFixed(2)) : 0;

  return {
    outputPixels,
    metrics: {
      totalPixels,
      changedPixelsCount: changedCount,
      maxChannelDiff: maxDiff,
      changedPercentage,
      carrierDimensions: { width: carrierW, height: carrierH },
      stegoDimensions: { width: stegoW, height: stegoH },
    },
  };
}

/**
 * 5-stop false-color thermal heatmap ramp:
 * Black -> Blue -> Cyan -> Yellow -> Red
 */
function heatMapRamp(t: number): [number, number, number] {
  if (t <= 0) return [10, 10, 15];
  if (t < 0.25) {
    const s = t / 0.25;
    return [0, Math.round(s * 100), Math.round(s * 255)];
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return [0, Math.round(100 + s * 155), Math.round(255 * (1 - s))];
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return [Math.round(s * 255), 255, 0];
  } else {
    const s = (t - 0.75) / 0.25;
    return [255, Math.round(255 * (1 - s)), 0];
  }
}
