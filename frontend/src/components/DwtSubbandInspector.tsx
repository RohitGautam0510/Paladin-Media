import React, { useState, useEffect, useRef } from 'react';
import { Layers, Crosshair, ZoomIn, Info } from 'lucide-react';
import { computeHaarHH, subbandToSpatialBlock, isValidSubbandCoord } from '../utils/differenceEngine';
import type { DiagnosticsMetadata } from './StegoInspectorModal';

interface DwtSubbandInspectorProps {
  dwtSubbands: Record<string, string> | null;
  modifiedHhImage: string | null;
  coverImageUrl: string | null;
  stegoImageUrl: string | null;
  diagnostics: DiagnosticsMetadata | null;
  activeRunId?: string | null;
}

export const DwtSubbandInspector: React.FC<DwtSubbandInspectorProps> = ({
  dwtSubbands,
  modifiedHhImage,
  coverImageUrl,
  stegoImageUrl,
  diagnostics,
  activeRunId,
}) => {
  const [selectedCoord, setSelectedCoord] = useState<{ r: number; c: number } | null>(null);
  const [hoverCoord, setHoverCoord] = useState<{ r: number; c: number } | null>(null);
  const [baseBand, setBaseBand] = useState<"modified" | "original">("modified");
  const [showEmbedded, setShowEmbedded] = useState<boolean>(true);
  const [showChanged, setShowChanged] = useState<boolean>(true);
  const [scale, setScale] = useState<number>(1); // 1 = 1x, 1.5 = 1.5x, 2 = 2x

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [coverPixelData, setCoverPixelData] = useState<Uint8ClampedArray | null>(null);
  const [stegoPixelData, setStegoPixelData] = useState<Uint8ClampedArray | null>(null);
  const [carrierDims, setCarrierDims] = useState<{ width: number; height: number }>({
    width: diagnostics?.carrier_dims ? diagnostics.carrier_dims[0] : 512,
    height: diagnostics?.carrier_dims ? diagnostics.carrier_dims[1] : 512,
  });

  // Synchronize carrier dimensions whenever diagnostics metadata updates
  useEffect(() => {
    if (diagnostics?.carrier_dims && diagnostics.carrier_dims[0] > 0 && diagnostics.carrier_dims[1] > 0) {
      setCarrierDims({ width: diagnostics.carrier_dims[0], height: diagnostics.carrier_dims[1] });
    }
  }, [diagnostics]);

  const subbandWidth = diagnostics?.subband_width || (diagnostics && typeof diagnostics.subband_dim === "object" ? diagnostics.subband_dim[1] : (typeof diagnostics?.subband_dim === "number" ? diagnostics.subband_dim : Math.floor(carrierDims.width / 2)));
  const subbandHeight = diagnostics?.subband_height || (diagnostics && typeof diagnostics.subband_dim === "object" ? diagnostics.subband_dim[0] : (typeof diagnostics?.subband_dim === "number" ? diagnostics.subband_dim : Math.floor(carrierDims.height / 2)));

  // Embedded and changed position sets for fast lookup
  const embeddedSet = useRef<Set<string>>(new Set());
  const changedSet = useRef<Set<string>>(new Set());

  useEffect(() => {
    embeddedSet.current.clear();
    changedSet.current.clear();

    if (diagnostics) {
      if (diagnostics.embedded_positions) {
        for (const [r, c] of diagnostics.embedded_positions) {
          embeddedSet.current.add(`${r},${c}`);
        }
      }
      if (diagnostics.changed_positions) {
        for (const [r, c] of diagnostics.changed_positions) {
          changedSet.current.add(`${r},${c}`);
        }
      }
    }
  }, [diagnostics]);

  // Decode carrier pixels independently so carrier dimensions and single-coefficient Haar formula work immediately
  useEffect(() => {
    if (!coverImageUrl) {
      setCoverPixelData(null);
      return;
    }

    let isSubscribed = true;
    const cov = new Image();
    cov.crossOrigin = "anonymous";
    cov.onload = () => {
      if (!isSubscribed) return;
      const cW = cov.naturalWidth;
      const cH = cov.naturalHeight;
      if (cW > 0 && cH > 0) {
        setCarrierDims({ width: cW, height: cH });
      }
      const off = document.createElement("canvas");
      off.width = cW;
      off.height = cH;
      const ctx = off.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(cov, 0, 0, cW, cH);
      setCoverPixelData(new Uint8ClampedArray(ctx.getImageData(0, 0, cW, cH).data));
    };
    cov.src = coverImageUrl;

    return () => {
      isSubscribed = false;
    };
  }, [coverImageUrl]);

  // Decode stego pixels independently whenever stego container is ready
  useEffect(() => {
    if (!stegoImageUrl) {
      setStegoPixelData(null);
      return;
    }

    let isSubscribed = true;
    const stg = new Image();
    stg.crossOrigin = "anonymous";
    stg.onload = () => {
      if (!isSubscribed) return;
      const sW = stg.naturalWidth;
      const sH = stg.naturalHeight;
      const off = document.createElement("canvas");
      off.width = sW;
      off.height = sH;
      const ctx = off.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(stg, 0, 0, sW, sH);
      setStegoPixelData(new Uint8ClampedArray(ctx.getImageData(0, 0, sW, sH).data));
    };
    stg.src = stegoImageUrl;

    return () => {
      isSubscribed = false;
    };
  }, [stegoImageUrl]);

  // Draw interactive subband canvas with overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const activeImageSrc = baseBand === "modified"
      ? modifiedHhImage
      : (dwtSubbands ? dwtSubbands["HH"] : null);

    if (!activeImageSrc) {
      ctx.clearRect(0, 0, subbandWidth, subbandHeight);
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Draw base display-normalized subband image
      ctx.drawImage(img, 0, 0, subbandWidth, subbandHeight);

      // Enforce image-boundary clipping within subband bounds
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, subbandWidth, subbandHeight);
      ctx.clip();

      // Layer 3: Overlay embedded positions in Red
      if (showEmbedded && diagnostics?.embedded_positions) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.7)"; // Red-500
        for (const [r, c] of diagnostics.embedded_positions) {
          if (!isValidSubbandCoord(r, c, subbandHeight, subbandWidth)) continue;
          ctx.fillRect(c, r, 1, 1);
        }
      }

      // Layer 4: Overlay changed positions in Cyan
      if (showChanged && diagnostics?.changed_positions) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.85)"; // Cyan-500
        for (const [r, c] of diagnostics.changed_positions) {
          if (!isValidSubbandCoord(r, c, subbandHeight, subbandWidth)) continue;
          ctx.fillRect(c, r, 1, 1);
        }
      }
      ctx.restore();

      // Highlight selected coordinate with crosshair
      const coordToMark = selectedCoord || hoverCoord;
      if (coordToMark) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.strokeRect(coordToMark.c - 2.5, coordToMark.r - 2.5, 5, 5);
      }
    };
    img.src = activeImageSrc;
  }, [baseBand, modifiedHhImage, dwtSubbands, showEmbedded, showChanged, diagnostics, selectedCoord, hoverCoord]);

  // Canvas click & mousemove handlers to determine subband (r, c)
  const getSubbandCoordFromEvent = (e: React.MouseEvent<HTMLCanvasElement>): { r: number; c: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * subbandWidth);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * subbandHeight);
    if (x < 0 || x >= subbandWidth || y < 0 || y >= subbandHeight) return null;
    return { r: y, c: x };
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coord = getSubbandCoordFromEvent(e);
    if (coord) setSelectedCoord(coord);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coord = getSubbandCoordFromEvent(e);
    setHoverCoord(coord);
  };

  const activeCoord = selectedCoord || hoverCoord;

  // Compute exact coefficient data for inspected coordinate
  let origVal: number | null = null;
  let modVal: number | null = null;
  let deltaVal: number | null = null;
  let isAssignedBit = false;
  let isValueChanged = false;
  let spatialBlock = null;

  if (activeCoord) {
    const key = `${activeCoord.r},${activeCoord.c}`;
    isAssignedBit = embeddedSet.current.has(key);
    isValueChanged = changedSet.current.has(key);
    spatialBlock = subbandToSpatialBlock(activeCoord.r, activeCoord.c);

    if (coverPixelData) {
      origVal = computeHaarHH(coverPixelData, carrierDims.width, activeCoord.r, activeCoord.c, 0);
    }
    if (stegoPixelData) {
      modVal = computeHaarHH(stegoPixelData, carrierDims.width, activeCoord.r, activeCoord.c, 0);
    }
    if (origVal !== null && modVal !== null) {
      deltaVal = modVal - origVal;
    }
  }

  return (
    <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm flex flex-col space-y-4">

      {/* HEADER & CONTROLS */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#212121] pb-3">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-primary" />
          <h2 className="text-sm font-semibold tracking-wide text-muted uppercase font-mono">
            DWT Subband Transform Inspection (Actual Coefficients)
          </h2>
          <span className="text-[10px] bg-primary/20 text-primary border border-primary/30 px-1.5 py-0.5 rounded font-mono font-bold">
            COEFFICIENT DOMAIN ({subbandWidth}×{subbandHeight})
          </span>
          {activeRunId && (
            <span className="text-[9px] bg-[#1a1a1a] text-muted border border-[#333] px-1.5 py-0.5 rounded font-mono truncate max-w-[140px]" title={activeRunId}>
              RUN: {activeRunId}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="flex items-center bg-[#0a0a0a] rounded border border-[#2a2a2a] p-0.5">
            <button
              onClick={() => setBaseBand("modified")}
              className={`px-2.5 py-1 rounded text-[11px] font-bold ${
                baseBand === "modified" ? "bg-primary text-white" : "text-muted hover:text-[#ededed]"
              }`}
            >
              Modified HH'
            </button>
            <button
              onClick={() => setBaseBand("original")}
              className={`px-2.5 py-1 rounded text-[11px] font-bold ${
                baseBand === "original" ? "bg-[#333] text-white" : "text-muted hover:text-[#ededed]"
              }`}
            >
              Original HH
            </button>
          </div>

          <button
            onClick={() => setScale(scale === 1 ? 1.5 : scale === 1.5 ? 2 : 1)}
            className="flex items-center bg-[#0a0a0a] border border-[#2a2a2a] px-2 py-1 rounded text-[11px] text-muted hover:text-[#ededed]"
            title="Toggle magnification"
          >
            <ZoomIn className="w-3 h-3 mr-1 text-primary" />
            <span>{scale}× View</span>
          </button>
        </div>
      </div>

      {/* 4 REGULAR SUBBAND THUMBNAILS (LL, LH, HL, HH) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
        {["LL", "LH", "HL", "HH"].map((band) => (
          <div key={band} className="border border-[#212121] rounded bg-[#0a0a0a] p-2 flex flex-col items-center">
            <div className="flex justify-between w-full text-[10px] font-mono text-muted mb-1">
              <span className="font-semibold">Subband: {band}</span>
              <span>{subbandWidth}×{subbandHeight}</span>
            </div>
            <div className="w-full aspect-square bg-[#050505] rounded border border-[#1a1a1a] flex items-center justify-center overflow-hidden">
              {dwtSubbands && dwtSubbands[band] ? (
                <img src={dwtSubbands[band]} alt={band} className="w-full h-full object-cover" />
              ) : (
                <span className="text-[10px] font-mono text-muted/60 text-center px-1">
                  {diagnostics?.completeness_status === "PARTIAL" || diagnostics?.coordinates_truncated
                    ? "VISUAL BOUNDED"
                    : "UNAVAILABLE"}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* INTERACTIVE TRANSFORM INSPECTOR (MODIFIED HH') */}
      <div className="border border-primary/30 rounded-lg bg-[#0a0a0a] p-4 flex flex-col xl:flex-row gap-5 items-start">

        {/* INTERACTIVE 256x256 CANVAS */}
        <div className="flex flex-col items-center flex-shrink-0">
          <div className="flex justify-between w-full text-xs font-mono mb-2">
            <span className="font-bold text-primary flex items-center">
              <Crosshair className="w-3.5 h-3.5 mr-1" />
              Interactive Subband Canvas
            </span>
            <span className="text-[10px] text-muted">Click/Hover to inspect</span>
          </div>

          <div
            className="border border-[#333] rounded overflow-hidden bg-[#050505] relative shadow-md"
            style={{
              width: Math.min(384, subbandWidth) * scale,
              height: Math.min(384, subbandHeight) * scale,
            }}
          >
            {modifiedHhImage ? (
              <canvas
                ref={canvasRef}
                width={subbandWidth}
                height={subbandHeight}
                onClick={handleCanvasClick}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoverCoord(null)}
                className="w-full h-full cursor-crosshair image-pixelated"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-xs font-mono text-muted/50">
                Awaiting transform embedding...
              </div>
            )}
          </div>

          {/* OVERLAY TOGGLES & LEGEND */}
          <div className="flex items-center space-x-4 mt-2.5 text-xs font-mono">
            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showEmbedded}
                onChange={(e) => setShowEmbedded(e.target.checked)}
                className="accent-red-500 rounded"
              />
              <span className="text-red-400 text-[11px] flex items-center">
                <span className="w-2 h-2 rounded-full bg-red-500 mr-1"></span>
                Embedded Positions
              </span>
            </label>
            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showChanged}
                onChange={(e) => setShowChanged(e.target.checked)}
                className="accent-cyan-500 rounded"
              />
              <span className="text-cyan-400 text-[11px] flex items-center">
                <span className="w-2 h-2 rounded-full bg-cyan-400 mr-1"></span>
                Changed Coefficients
              </span>
            </label>
          </div>
        </div>

        {/* COEFFICIENT VALUE DETAILS PANEL */}
        <div className="flex-1 w-full bg-[#0d0d0d] border border-[#262626] rounded-lg p-4 flex flex-col justify-between font-mono text-xs">
          <div>
            <div className="flex justify-between items-center pb-2 border-b border-[#222] mb-3">
              <span className="text-xs font-bold text-[#ededed] flex items-center">
                <Info className="w-3.5 h-3.5 mr-1 text-primary" />
                COEFFICIENT TELEMETRY INSPECTOR
              </span>
              {activeCoord && (
                <span className="text-[10px] bg-primary/20 text-primary border border-primary/30 px-2 py-0.5 rounded font-bold">
                  Row: {activeCoord.r}, Col: {activeCoord.c}
                </span>
              )}
            </div>

            {activeCoord ? (
              <div className="space-y-2.5">
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-[#141414] p-2 rounded border border-[#242424]">
                    <span className="text-muted block text-[10px] uppercase">Original HH Value</span>
                    <span className="text-[#ededed] font-bold text-sm">
                      {origVal !== null ? origVal.toFixed(2) : "N/A"}
                    </span>
                  </div>
                  <div className="bg-[#141414] p-2 rounded border border-[#242424]">
                    <span className="text-muted block text-[10px] uppercase">Modified HH' Value</span>
                    <span className="text-primary font-bold text-sm">
                      {modVal !== null ? modVal.toFixed(2) : "N/A"}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-[#141414] p-2 rounded border border-[#242424]">
                    <span className="text-muted block text-[10px] uppercase">Numerical Delta (Δ)</span>
                    <span className={`font-bold text-sm ${
                      isValueChanged ? "text-cyan-400" : "text-muted"
                    }`}>
                      {deltaVal !== null ? (deltaVal > 0 ? `+${deltaVal.toFixed(2)}` : deltaVal.toFixed(2)) : "0.00"}
                      {isValueChanged ? " (CHANGED)" : ""}
                    </span>
                  </div>
                  <div className="bg-[#141414] p-2 rounded border border-[#242424]">
                    <span className="text-muted block text-[10px] uppercase">Assigned to Bit</span>
                    <span className={`font-bold text-sm ${isAssignedBit ? "text-red-400" : "text-muted"}`}>
                      {isAssignedBit ? "YES (CARRIER MASK)" : "NO"}
                    </span>
                  </div>
                </div>

                <div className="bg-[#141414] p-2.5 rounded border border-[#242424] text-[11px]">
                  <span className="text-muted block text-[10px] uppercase mb-1">
                    Projected 2×2 Carrier Spatial Support
                  </span>
                  {spatialBlock ? (
                    <div className="text-[#ededed]">
                      Carrier Pixels: X: [{spatialBlock.minX} .. {spatialBlock.maxX}], Y: [{spatialBlock.minY} .. {spatialBlock.maxY}]
                    </div>
                  ) : (
                    <span className="text-muted">N/A</span>
                  )}
                  <p className="text-[10px] text-muted mt-1 leading-normal">
                    In 1-level 2D Haar DWT, one HH coefficient maps mathematically to a 2×2 spatial pixel block.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-muted/60 text-center py-8">
                Click or hover anywhere on the {subbandWidth}×{subbandHeight} canvas to inspect individual coefficient coordinates and their mathematical support.
              </div>
            )}
          </div>

          <div className="pt-3 mt-3 border-t border-[#222] text-[10px] text-muted">
            <span className="text-primary font-semibold">Transform Domain Invariant:</span> {carrierDims.width}×{carrierDims.height} RGB carrier produces {subbandWidth}×{subbandHeight} subbands via 1-level 2D Haar transform.
          </div>
        </div>

      </div>

    </div>
  );
};
