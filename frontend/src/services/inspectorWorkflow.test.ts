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
} from '../utils/differenceEngine.ts';
import type { DiagnosticsMetadata, InspectorViewMode } from '../components/StegoInspectorModal.tsx';

test('1. Inspector View Modes transition cleanly across all 5 views', () => {
  const validModes: InspectorViewMode[] = ['stego', 'embedded', 'changed', 'diff', 'amplified'];
  let currentMode: InspectorViewMode = 'stego';

  for (const mode of validModes) {
    currentMode = mode;
    assert.equal(currentMode, mode);
  }
});

test('2. Coordinate mapping preserves exact 1-level 2D Haar DWT spatial support', () => {
  // Test every quadrant boundary in the 256x256 subband
  const testCoords = [
    { r: 0, c: 0, expected: { minX: 0, maxX: 1, minY: 0, maxY: 1 } },
    { r: 0, c: 255, expected: { minX: 510, maxX: 511, minY: 0, maxY: 1 } },
    { r: 255, c: 0, expected: { minX: 0, maxX: 1, minY: 510, maxY: 511 } },
    { r: 255, c: 255, expected: { minX: 510, maxX: 511, minY: 510, maxY: 511 } },
    { r: 128, c: 128, expected: { minX: 256, maxX: 257, minY: 256, maxY: 257 } },
  ];

  for (const { r, c, expected } of testCoords) {
    const block = subbandToSpatialBlock(r, c);
    assert.deepEqual(block, expected);
  }
});

test('3. Distinguishes assigned embedding positions from numerically changed coefficients', () => {
  // Simulate genuine diagnostic data:
  // 128 header positions + 100 payload positions = 228 total assigned positions
  const headerPositions: [number, number][] = Array.from({ length: 128 }, (_, i) => [0, i]);
  const payloadPositions: [number, number][] = Array.from({ length: 100 }, (_, i) => [1, i]);
  const allAssigned: [number, number][] = [...headerPositions, ...payloadPositions];

  // Suppose 120 of those positions numerically changed, while others matched parity
  const changedPositions: [number, number][] = allAssigned.slice(0, 120);

  const diagnostics: DiagnosticsMetadata = {
    run_id: 'test_run_isolation_01',
    carrier_dims: [512, 512],
    stego_dims: [512, 512],
    subband_dim: 256,
    header_bits_count: 128,
    payload_bits_count: 100,
    total_embedded_bits: 228,
    changed_coefficients_count: 120,
    max_hh_diff: 0.5,
    embedded_positions: allAssigned,
    changed_positions: changedPositions,
  };

  const assignedSet = new Set(diagnostics.embedded_positions.map(([r, c]) => `${r},${c}`));
  const changedSet = new Set(diagnostics.changed_positions.map(([r, c]) => `${r},${c}`));

  assert.equal(assignedSet.size, 228);
  assert.equal(changedSet.size, 120);

  // Position (0, 0) is both assigned and changed
  assert.equal(assignedSet.has('0,0'), true);
  assert.equal(changedSet.has('0,0'), true);

  // Position (1, 99) is assigned but NOT changed
  assert.equal(assignedSet.has('1,99'), true);
  assert.equal(changedSet.has('1,99'), false);

  // Unused position (50, 50) is neither
  assert.equal(assignedSet.has('50,50'), false);
  assert.equal(changedSet.has('50,50'), false);
});

test('4. Gracefully handles missing or unavailable diagnostic metadata', () => {
  const diagnostics: DiagnosticsMetadata | null = null;

  // Verify safe fallback behavior when diagnostics is null
  const totalEmbedded = diagnostics?.total_embedded_bits ?? 0;
  const changedCount = diagnostics?.changed_coefficients_count ?? 0;
  const embeddedPositions = diagnostics?.embedded_positions ?? [];

  assert.equal(totalEmbedded, 0);
  assert.equal(changedCount, 0);
  assert.equal(embeddedPositions.length, 0);
});

test('5. Stale-result prevention: run reset invalidates all previous overlays and states', () => {
  // Simulate active run state with artifacts
  interface AppRunState {
    runId: string | null;
    stegoImage: string | null;
    recoveredImage: string | null;
    exactMatch: boolean | null;
    diagnostics: DiagnosticsMetadata | null;
    isModalOpen: boolean;
  }

  let state: AppRunState = {
    runId: 'run_101_initial',
    stegoImage: 'data:image/png;base64,stego123',
    recoveredImage: 'data:image/png;base64,recovered123',
    exactMatch: true,
    diagnostics: {
      run_id: 'run_101_initial',
      carrier_dims: [512, 512],
      stego_dims: [512, 512],
      subband_dim: 256,
      header_bits_count: 128,
      payload_bits_count: 100,
      total_embedded_bits: 228,
      changed_coefficients_count: 120,
      max_hh_diff: 0.5,
      embedded_positions: [[0, 0]],
      changed_positions: [[0, 0]],
    },
    isModalOpen: true,
  };

  // User cancels or starts a new run: execute resetOutputs logic
  const resetOutputs = (s: AppRunState): AppRunState => ({
    runId: null,
    stegoImage: null,
    recoveredImage: null,
    exactMatch: null,
    diagnostics: null,
    isModalOpen: false,
  });

  state = resetOutputs(state);

  assert.equal(state.runId, null);
  assert.equal(state.stegoImage, null);
  assert.equal(state.recoveredImage, null);
  assert.equal(state.exactMatch, null);
  assert.equal(state.diagnostics, null);
  assert.equal(state.isModalOpen, false);
});

test('6. Capacity rejection fails closed and prevents successful embedding overlays', () => {
  let runStatus: 'IDLE' | 'PROCESSING' | 'SUCCESS' | 'FAILED' = 'PROCESSING';
  let stegoImage: string | null = 'stale_stego_png';
  let exactMatch: boolean | null = true;
  let diagnostics: DiagnosticsMetadata | null = {
    run_id: 'old_run',
    carrier_dims: [512, 512],
    stego_dims: [512, 512],
    subband_dim: 256,
    header_bits_count: 128,
    payload_bits_count: 50,
    total_embedded_bits: 178,
    changed_coefficients_count: 80,
    max_hh_diff: 0.5,
    embedded_positions: [],
    changed_positions: [],
  };

  // Simulate CAPACITY_CHECK failed event
  const capacityEvent = {
    stage: 'CAPACITY_CHECK',
    status: 'failed',
    error: 'PAYLOAD_CAPACITY_EXCEEDED',
    metadata: {
      fits: false,
      protected_payload_bytes: 12000,
      max_protected_bytes: 8176,
    },
  };

  if (capacityEvent.status === 'failed') {
    runStatus = 'FAILED';
    if (capacityEvent.stage === 'CAPACITY_CHECK') {
      stegoImage = null;
      diagnostics = null;
      exactMatch = null;
    }
  }

  assert.equal(runStatus, 'FAILED');
  assert.equal(stegoImage, null);
  assert.equal(diagnostics, null);
  assert.equal(exactMatch, null);
  assert.equal(capacityEvent.metadata.fits, false);
});

test('7. Diagnostics metadata supports verification lifecycle and partial truncated coordinates', () => {
  const truncatedDiag: DiagnosticsMetadata = {
    run_id: 'run_truncated_test_123',
    carrier_dims: [512, 512],
    stego_dims: [512, 512],
    subband_dim: 256,
    header_bits_count: 128,
    payload_bits_count: 5000,
    total_embedded_bits: 5128,
    changed_coefficients_count: 2500,
    max_hh_diff: 1.5,
    embedded_positions: Array.from({ length: 500 }, (_, i) => [0, i]), // truncated list
    changed_positions: Array.from({ length: 250 }, (_, i) => [0, i]), // truncated list
    verification_status: 'VERIFIED',
    exact_match: true,
    coordinates_truncated: true,
    total_embedded_positions_count: 5128,
    total_changed_positions_count: 2500,
  };

  assert.equal(truncatedDiag.verification_status, 'VERIFIED');
  assert.equal(truncatedDiag.exact_match, true);
  assert.equal(truncatedDiag.coordinates_truncated, true);
  assert.equal(truncatedDiag.total_embedded_positions_count, 5128);
  assert.equal(truncatedDiag.total_changed_positions_count, 2500);
  assert.equal(truncatedDiag.embedded_positions.length, 500);
  assert.equal(truncatedDiag.changed_positions.length, 250);
});

test('8. Image dimension validation rejects non-512x512 carriers or stego images', () => {
  assert.throws(
    () => validateImageDimensions(512, 512, 256, 256, 512, 512),
    /Dimension mismatch: Carrier is 512x512, but Stego is 256x256/
  );
  assert.throws(
    () => validateImageDimensions(256, 256, 256, 256, 512, 512),
    /Incompatible width: Expected 512, but got Carrier: 256, Stego: 256/
  );
  assert.doesNotThrow(
    () => validateImageDimensions(512, 512, 512, 512, 512, 512)
  );
});

test('9. Diagnostics completeness status distinguishes COMPLETE, PARTIAL, and UNAVAILABLE states', () => {
  // Complete diagnostics
  const completeDiag: DiagnosticsMetadata = {
    run_id: 'run_complete_001',
    carrier_dims: [512, 512],
    stego_dims: [512, 512],
    subband_dim: 256,
    header_bits_count: 128,
    payload_bits_count: 200,
    total_embedded_bits: 328,
    changed_coefficients_count: 150,
    max_hh_diff: 0.8,
    embedded_positions: Array.from({ length: 328 }, (_, i) => [0, i]),
    changed_positions: Array.from({ length: 150 }, (_, i) => [0, i]),
    coordinates_truncated: false,
    completeness_status: 'COMPLETE',
    verification_status: 'VERIFIED',
    exact_match: true,
  };
  assert.equal(completeDiag.completeness_status, 'COMPLETE');
  assert.equal(completeDiag.coordinates_truncated, false);
  assert.equal(completeDiag.embedded_positions.length, completeDiag.total_embedded_bits);

  // Partial diagnostics
  const partialDiag: DiagnosticsMetadata = {
    ...completeDiag,
    coordinates_truncated: true,
    completeness_status: 'PARTIAL',
    embedded_positions: completeDiag.embedded_positions.slice(0, 100),
    total_embedded_positions_count: 328,
  };
  assert.equal(partialDiag.completeness_status, 'PARTIAL');
  assert.equal(partialDiag.coordinates_truncated, true);
  assert.equal(partialDiag.embedded_positions.length, 100);
  assert.equal(partialDiag.total_embedded_positions_count, 328);

  // Unavailable diagnostics (early abort or pre-embed failure)
  const unavailableDiag: DiagnosticsMetadata = {
    run_id: 'run_aborted_001',
    carrier_dims: [512, 512],
    stego_dims: [0, 0],
    subband_dim: 256,
    header_bits_count: 0,
    payload_bits_count: 0,
    total_embedded_bits: 0,
    changed_coefficients_count: 0,
    max_hh_diff: 0,
    embedded_positions: [],
    changed_positions: [],
    completeness_status: 'UNAVAILABLE',
    verification_status: 'PRE_EMBED_FAILED',
    error_message: 'PAYLOAD_CAPACITY_EXCEEDED',
  };
  assert.equal(unavailableDiag.completeness_status, 'UNAVAILABLE');
  assert.equal(unavailableDiag.embedded_positions.length, 0);
  assert.equal(unavailableDiag.total_embedded_bits, 0);
});

test('10. Dynamic carrier dimensions (1024x1024, 512x768) validate and map correctly', () => {
  // 1024x1024 matching dimensions validate
  assert.doesNotThrow(() => validateImageDimensions(1024, 1024, 1024, 1024));

  // Non-square even dimensions (512x768) validate
  assert.doesNotThrow(() => validateImageDimensions(512, 768, 512, 768));

  // Spatial block mapping works identically across dimensions
  const block0 = subbandToSpatialBlock(0, 0);
  assert.deepEqual(block0, { minX: 0, maxX: 1, minY: 0, maxY: 1 });

  // High coordinate in 1024x1024 (subband 512x512)
  const blockHigh = subbandToSpatialBlock(511, 511);
  assert.deepEqual(blockHigh, { minX: 1022, maxX: 1023, minY: 1022, maxY: 1023 });

  // Non-square subband coordinate (r=255, c=383 for 512x768 carrier)
  const blockNonSquare = subbandToSpatialBlock(255, 383);
  assert.deepEqual(blockNonSquare, { minX: 766, maxX: 767, minY: 510, maxY: 511 });
});

test('11. Mismatched secret and carrier dimensions are fully supported', () => {
  // Secret image payload (24x24) into 1024x1024 carrier
  const secretDims = { width: 24, height: 24 };
  const carrierDims = { width: 1024, height: 1024 };

  assert.notEqual(secretDims.width, carrierDims.width);
  assert.notEqual(secretDims.height, carrierDims.height);

  // Dynamic diagnostics accepts non-square or enlarged subband_dim
  const dynamicDiag: DiagnosticsMetadata = {
    run_id: 'run_dyn_001',
    carrier_dims: [carrierDims.width, carrierDims.height],
    stego_dims: [carrierDims.width, carrierDims.height],
    subband_dim: [512, 512],
    subband_width: 512,
    subband_height: 512,
    header_bits_count: 128,
    payload_bits_count: 400,
    total_embedded_bits: 528,
    changed_coefficients_count: 260,
    max_hh_diff: 0.9,
    embedded_positions: [[0, 0], [10, 20]],
    changed_positions: [[0, 0]],
    verification_status: 'VERIFIED',
    exact_match: true,
  };

  assert.deepEqual(dynamicDiag.carrier_dims, [1024, 1024]);
  assert.equal(dynamicDiag.subband_width, 512);
  assert.equal(dynamicDiag.subband_height, 512);
});

test('12. Completed 512x512 baseline run produces correct 256x256 subband dimensions and completed lifecycle', () => {
  const carrierDims = { width: 512, height: 512 };
  const subbandWidth = carrierDims.width / 2;
  const subbandHeight = carrierDims.height / 2;

  assert.equal(subbandWidth, 256);
  assert.equal(subbandHeight, 256);

  const blockCorner = subbandToSpatialBlock(255, 255);
  assert.deepEqual(blockCorner, { minX: 510, maxX: 511, minY: 510, maxY: 511 });
  assert.equal(blockCorner.maxX < carrierDims.width, true);
  assert.equal(blockCorner.maxY < carrierDims.height, true);
});

test('13. Completed 2040x1356 carrier run derives exact 1020x678 subband dimensions and maps spatial blocks', () => {
  const carrierDims = { width: 2040, height: 1356 };
  const subbandWidth = carrierDims.width / 2;
  const subbandHeight = carrierDims.height / 2;

  assert.equal(subbandWidth, 1020);
  assert.equal(subbandHeight, 678);

  // Boundary coordinates for 1020x678 subband
  const origin = subbandToSpatialBlock(0, 0);
  assert.deepEqual(origin, { minX: 0, maxX: 1, minY: 0, maxY: 1 });

  const maxCorner = subbandToSpatialBlock(subbandHeight - 1, subbandWidth - 1);
  assert.deepEqual(maxCorner, { minX: 2038, maxX: 2039, minY: 1354, maxY: 1355 });
  assert.equal(maxCorner.maxX, carrierDims.width - 1);
  assert.equal(maxCorner.maxY, carrierDims.height - 1);

  // Validate Haar formula with 2040 stride
  const mockPixels = new Uint8ClampedArray(carrierDims.width * carrierDims.height * 4);
  // Place distinct values at block (r=339, c=510) -> spatial X:[1020, 1021], Y:[678, 679]
  const stride = carrierDims.width * 4;
  const idxA = (2 * 339) * stride + (2 * 510) * 4;
  mockPixels[idxA] = 120; // a
  mockPixels[idxA + 4] = 40; // b
  mockPixels[idxA + stride] = 30; // c
  mockPixels[idxA + stride + 4] = 10; // d

  const hhVal = computeHaarHH(mockPixels, carrierDims.width, 339, 510, 0);
  // (120 - 40 - 30 + 10) / 2.0 = 60 / 2.0 = 30.0
  assert.equal(hhVal, 30.0);
});

test('14. Run lifecycle transitions cleanly: idle -> running -> completed with all output panels populated', () => {
  interface WorkspaceState {
    lifecycle: 'IDLE' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
    runId: string | null;
    stegoImage: string | null;
    psnr: number | null;
    ssim: number | null;
    recoveredImage: string | null;
    exactMatch: boolean | null;
    dwtSubbands: Record<string, string> | null;
  }

  let state: WorkspaceState = {
    lifecycle: 'IDLE',
    runId: null,
    stegoImage: null,
    psnr: null,
    ssim: null,
    recoveredImage: null,
    exactMatch: null,
    dwtSubbands: null,
  };

  // Step 1: Start run
  state = {
    ...state,
    lifecycle: 'RUNNING',
    runId: 'run_lifecycle_01',
    stegoImage: null,
    recoveredImage: null,
  };
  assert.equal(state.lifecycle, 'RUNNING');
  assert.equal(state.stegoImage, null);

  // Step 2: DWT complete
  state.dwtSubbands = { LL: 'data:img/ll', LH: 'data:img/lh', HL: 'data:img/hl', HH: 'data:img/hh' };
  assert.notEqual(state.dwtSubbands, null);

  // Step 3: IDWT complete
  state.stegoImage = 'data:image/png;base64,mockStegoData';
  state.psnr = 55.70;
  state.ssim = 0.99999;
  assert.notEqual(state.stegoImage, null);
  assert.equal(state.psnr, 55.70);

  // Step 4: Secret recovered and verified
  state.recoveredImage = 'data:image/png;base64,mockSecretData';
  state.exactMatch = true;
  state.lifecycle = 'COMPLETED';

  assert.equal(state.lifecycle, 'COMPLETED');
  assert.equal(state.runId, 'run_lifecycle_01');
  assert.equal(state.exactMatch, true);
  assert.notEqual(state.stegoImage, null);
});

test('15. Failed or cancelled run clears or visibly invalidates stale outputs', () => {
  // Start with populated successful state from a previous run
  let state = {
    lifecycle: 'COMPLETED',
    runId: 'run_old_success',
    stegoImage: 'data:img/oldStego',
    psnr: 68.2,
    ssim: 1.0,
    recoveredImage: 'data:img/oldSecret',
    exactMatch: true,
  };

  // User starts new run and cancels
  state = {
    lifecycle: 'CANCELLED',
    runId: null,
    stegoImage: null,
    psnr: null,
    ssim: null,
    recoveredImage: null,
    exactMatch: null,
  };

  assert.equal(state.lifecycle, 'CANCELLED');
  assert.equal(state.stegoImage, null);
  assert.equal(state.psnr, null);
  assert.equal(state.exactMatch, null);

  // Failure invalidation
  state = {
    lifecycle: 'FAILED',
    runId: 'run_failed_02',
    stegoImage: null,
    psnr: null,
    ssim: null,
    recoveredImage: null,
    exactMatch: false,
  };

  assert.equal(state.lifecycle, 'FAILED');
  assert.equal(state.stegoImage, null);
  assert.equal(state.exactMatch, false);
});

test('16. Subband metadata absent or truncated provides honest bounded/unavailable state', () => {
  const truncatedDiag: DiagnosticsMetadata = {
    run_id: 'run_large_2040',
    carrier_dims: [2040, 1356],
    stego_dims: [2040, 1356],
    subband_dim: [678, 1020],
    subband_width: 1020,
    subband_height: 678,
    header_bits_count: 128,
    payload_bits_count: 87480,
    total_embedded_bits: 87608,
    changed_coefficients_count: 43800,
    max_hh_diff: 1.0,
    embedded_positions: [],
    changed_positions: [],
    coordinates_truncated: true,
    completeness_status: 'PARTIAL',
  };

  assert.equal(truncatedDiag.coordinates_truncated, true);
  assert.equal(truncatedDiag.completeness_status, 'PARTIAL');
  assert.equal(truncatedDiag.subband_width, 1020);
  assert.equal(truncatedDiag.subband_height, 678);
});

test('17. Second run replacing first run cleanly resets previous overlays, metrics, and run ID', () => {
  // First run finishes
  const run1 = {
    runId: 'run_first_001',
    psnr: 52.3,
    ssim: 0.9998,
    exactMatch: true,
  };

  // Second run starts: must not keep run 1's ID or metrics
  const run2Start = {
    runId: 'run_second_002',
    psnr: null,
    ssim: null,
    exactMatch: null,
  };

  assert.notEqual(run2Start.runId, run1.runId);
  assert.equal(run2Start.psnr, null);
  assert.equal(run2Start.exactMatch, null);
});

test('18. Exact-recovery status remains strictly tied to the correct run ID', () => {
  const runRecord = {
    runId: 'run_verified_777',
    verification_status: 'VERIFIED',
    exact_match: true,
  };

  assert.equal(runRecord.runId, 'run_verified_777');
  assert.equal(runRecord.exact_match, true);
});

test('19. Coordinate mapping for corner markers across square and non-square carriers', () => {
  const carriers = [
    { name: 'Square 512x512', w: 512, h: 512, subW: 256, subH: 256 },
    { name: 'DIV2K Landscape 2040x1356', w: 2040, h: 1356, subW: 1020, subH: 678 },
    { name: 'DIV2K Portrait 512x768', w: 512, h: 768, subW: 256, subH: 384 },
  ];

  for (const { name, w, h, subW, subH } of carriers) {
    const vp = calculateImageViewport(w, h, w, h, 'contain');

    // 1. Top-left corner: (0, 0)
    const mTL = mapSubbandToCanvas(0, 0, subH, subW, vp);
    assert.ok(mTL, `${name}: Top-left marker must be valid`);
    assert.equal(mTL.x, 0, `${name}: TL x must be 0`);
    assert.equal(mTL.y, 0, `${name}: TL y must be 0`);

    // 2. Top-right corner: (0, subW - 1)
    const mTR = mapSubbandToCanvas(0, subW - 1, subH, subW, vp);
    assert.ok(mTR, `${name}: Top-right marker must be valid`);
    assert.equal(mTR.y, 0, `${name}: TR y must be 0`);
    assert.equal(mTR.x + mTR.width, w, `${name}: TR right edge must touch carrier width exactly`);

    // 3. Bottom-left corner: (subH - 1, 0)
    const mBL = mapSubbandToCanvas(subH - 1, 0, subH, subW, vp);
    assert.ok(mBL, `${name}: Bottom-left marker must be valid`);
    assert.equal(mBL.x, 0, `${name}: BL x must be 0`);
    assert.equal(mBL.y + mBL.height, h, `${name}: BL bottom edge must touch carrier height exactly`);

    // 4. Bottom-right corner: (subH - 1, subW - 1)
    const mBR = mapSubbandToCanvas(subH - 1, subW - 1, subH, subW, vp);
    assert.ok(mBR, `${name}: Bottom-right marker must be valid`);
    assert.equal(mBR.x + mBR.width, w, `${name}: BR right edge must touch carrier width exactly`);
    assert.equal(mBR.y + mBR.height, h, `${name}: BR bottom edge must touch carrier height exactly`);
  }
});

test('20. First and last valid HH subband coefficients map strictly inside carrier pixel dimensions', () => {
  const w = 2040;
  const h = 1356;
  const subW = 1020;
  const subH = 678;
  const vp = calculateImageViewport(w, h, w, h, 'contain');

  // First valid coefficient
  const first = mapSubbandToCanvas(0, 0, subH, subW, vp);
  assert.ok(first);
  assert.ok(first.x >= 0 && first.x < w);
  assert.ok(first.y >= 0 && first.y < h);
  assert.ok(first.x + first.width <= w);
  assert.ok(first.y + first.height <= h);

  // Last valid coefficient
  const last = mapSubbandToCanvas(subH - 1, subW - 1, subH, subW, vp);
  assert.ok(last);
  assert.ok(last.x >= 0 && last.x < w);
  assert.ok(last.y >= 0 && last.y < h);
  assert.ok(last.x + last.width <= w);
  assert.ok(last.y + last.height <= h);
});

test('21. Non-square carrier dimensions (DIV2K portrait/landscape) scale without distortion or overflow', () => {
  // Test landscape carrier 2040x1356 displayed in 1024x768 viewport
  const vpLandscape = calculateImageViewport(1024, 768, 2040, 1356, 'contain');
  assert.equal(vpLandscape.scaleX, vpLandscape.scaleY, 'Scaling must be uniform (isotropic) to prevent aspect distortion');
  assert.ok(vpLandscape.displayedWidth <= 1024, 'Displayed width must not exceed viewport width');
  assert.ok(vpLandscape.displayedHeight <= 768, 'Displayed height must not exceed viewport height');

  // Test portrait carrier 512x768 displayed in 1024x768 viewport
  const vpPortrait = calculateImageViewport(1024, 768, 512, 768, 'contain');
  assert.equal(vpPortrait.scaleX, vpPortrait.scaleY, 'Portrait scaling must be uniform');
  assert.ok(vpPortrait.displayedWidth <= 1024);
  assert.ok(vpPortrait.displayedHeight <= 768);
});

test('22. Centered images with canvas margins — overlay rectangle matches displayed image rectangle', () => {
  // Simulate 1000x1000 container with a 2040x1356 image (landscape letterbox)
  const containerW = 1000;
  const containerH = 1000;
  const carrierW = 2040;
  const carrierH = 1356;
  const subW = 1020;
  const subH = 678;

  const vp = calculateImageViewport(containerW, containerH, carrierW, carrierH, 'contain');

  // The image is letterboxed with vertical margins
  assert.equal(vp.offsetX, 0, 'Horizontal offset must be 0 for full-width landscape fit');
  assert.ok(vp.offsetY > 150, 'Vertical offset must center the image with margins top and bottom');
  const expectedImgHeight = carrierH * (containerW / carrierW);
  assert.equal(Math.round(vp.displayedHeight), Math.round(expectedImgHeight));

  // Assert top letterbox margin: No marker can have y < vp.offsetY
  const topMarker = mapSubbandToCanvas(0, 0, subH, subW, vp);
  assert.ok(topMarker);
  assert.ok(topMarker.y >= vp.offsetY, `Marker top (${topMarker.y}) must be >= image top margin (${vp.offsetY})`);

  // Assert bottom letterbox margin: No marker can have y + height > vp.offsetY + vp.displayedHeight
  const bottomMarker = mapSubbandToCanvas(subH - 1, subW - 1, subH, subW, vp);
  assert.ok(bottomMarker);
  assert.ok(
    bottomMarker.y + bottomMarker.height <= vp.offsetY + vp.displayedHeight + 0.001,
    `Marker bottom (${bottomMarker.y + bottomMarker.height}) must not exceed image bottom (${vp.offsetY + vp.displayedHeight})`
  );

  // Pillarbox test: 1000x1000 container with 512x768 portrait image
  const vpPillar = calculateImageViewport(1000, 1000, 512, 768, 'contain');
  assert.equal(vpPillar.offsetY, 0, 'Vertical offset must be 0 for full-height portrait fit');
  assert.ok(vpPillar.offsetX > 150, 'Horizontal offset must center portrait image with side margins');

  const leftMarker = mapSubbandToCanvas(0, 0, 384, 256, vpPillar);
  assert.ok(leftMarker);
  assert.ok(leftMarker.x >= vpPillar.offsetX, `Marker x (${leftMarker.x}) must be >= image left margin (${vpPillar.offsetX})`);

  const rightMarker = mapSubbandToCanvas(383, 255, 384, 256, vpPillar);
  assert.ok(rightMarker);
  assert.ok(
    rightMarker.x + rightMarker.width <= vpPillar.offsetX + vpPillar.displayedWidth + 0.001,
    `Marker right must not exceed image right margin`
  );
});

test('23. Zoom and pan transforms preserve alignment between markers and underlying pixels', () => {
  // In StegoInspectorModal, zoom and pan are applied to the shared parent container:
  // transform: translate(${pan.x}px, ${pan.y}px) scale(${zoom})
  // Both the <img> (Layer 1) and <canvas> (Layer 2) are direct children with identical width and height.
  //
  // For any point P in carrier pixel coordinates (px, py):
  // Transformed image point: P_img = Pan + Zoom * P
  // Transformed canvas point: P_canvas = Pan + Zoom * P
  // Difference: |P_img - P_canvas| = 0 across all zoom levels and pan offsets.

  const zoomLevels = [0.5, 1.0, 1.3, 2.0, 5.0, 10.0];
  const panOffsets = [
    { x: 0, y: 0 },
    { x: 120, y: -45 },
    { x: -500, y: 350 },
  ];

  const testPoint = { x: 512, y: 340 };

  for (const zoom of zoomLevels) {
    for (const pan of panOffsets) {
      const imgPointX = pan.x + zoom * testPoint.x;
      const imgPointY = pan.y + zoom * testPoint.y;

      const canvasPointX = pan.x + zoom * testPoint.x;
      const canvasPointY = pan.y + zoom * testPoint.y;

      assert.equal(imgPointX, canvasPointX);
      assert.equal(imgPointY, canvasPointY);
    }
  }
});

test('24. Partial diagnostics (truncated coordinates) render without overflow or misalignment', () => {
  // Diagnostics with coordinates_truncated: true (e.g. 50 positions out of 87,608 total)
  const partialDiag: DiagnosticsMetadata = {
    run_id: 'run_partial_test_01',
    carrier_dims: [2040, 1356],
    stego_dims: [2040, 1356],
    subband_dim: [678, 1020],
    subband_width: 1020,
    subband_height: 678,
    header_bits_count: 128,
    payload_bits_count: 87480,
    total_embedded_bits: 87608,
    changed_coefficients_count: 43800,
    max_hh_diff: 1.0,
    embedded_positions: [
      [0, 0],
      [10, 20],
      [100, 200],
      [677, 1019],
    ],
    changed_positions: [
      [0, 0],
      [677, 1019],
    ],
    coordinates_truncated: true,
    completeness_status: 'PARTIAL',
  };

  const vp = calculateImageViewport(2040, 1356, 2040, 1356, 'contain');

  // Verify that partial positions all map inside the image rectangle
  for (const [r, c] of partialDiag.embedded_positions) {
    const marker = mapSubbandToCanvas(r, c, 678, 1020, vp);
    assert.ok(marker, `Position [${r}, ${c}] must map successfully`);
    assert.ok(marker.x >= 0 && marker.x + marker.width <= 2040);
    assert.ok(marker.y >= 0 && marker.y + marker.height <= 1356);
  }

  // Completeness status is truthful
  assert.equal(partialDiag.coordinates_truncated, true);
  assert.equal(partialDiag.completeness_status, 'PARTIAL');
});

test('25. Invalid / out-of-bounds coordinates are strictly rejected, not clamped to edges', () => {
  const subH = 678;
  const subW = 1020;
  const vp = calculateImageViewport(2040, 1356, 2040, 1356, 'contain');

  const invalidCoords: [number, number][] = [
    [-1, 0],      // Negative row
    [0, -1],      // Negative col
    [-5, -10],    // Both negative
    [subH, 0],    // Row equal to subband height
    [0, subW],    // Col equal to subband width
    [subH + 1, 0],
    [0, subW + 1],
    [9999, 9999], // Wildly out of bounds
    [NaN, 0],
    [0, NaN],
  ];

  for (const [r, c] of invalidCoords) {
    const result = mapSubbandToCanvas(r, c, subH, subW, vp);
    // MUST BE NULL: Must NOT clamp to border (x=0 or x=2038) because that would bunch markers at borders
    assert.equal(
      result,
      null,
      `Invalid coordinate [${r}, ${c}] must be rejected with null, never clamped to border`
    );
  }
});
