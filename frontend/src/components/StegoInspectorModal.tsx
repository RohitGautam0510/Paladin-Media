import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Activity,
  Sliders,
  Eye,
  EyeOff,
  Download,
  Info
} from 'lucide-react';
import {
  computePixelDifference,
  calculateImageViewport,
  mapSubbandToCanvas
} from '../utils/differenceEngine';

export type InspectorViewMode =
  | "stego"
  | "embedded"
  | "changed"
  | "diff"
  | "amplified";

export interface DiagnosticsMetadata {
  run_id: string;
  carrier_dims: [number, number];
  stego_dims: [number, number];
  subband_dim: number | [number, number];
  subband_width?: number;
  subband_height?: number;
  header_bits_count: number;
  payload_bits_count: number;
  total_embedded_bits: number;
  changed_coefficients_count: number;
  max_hh_diff: number;
  embedded_positions: [number, number][];
  changed_positions: [number, number][];
  verification_status?: string; // PENDING | VERIFIED | FAILED | ABORTED | PRE_EMBED_FAILED
  exact_match?: boolean | null;
  coordinates_truncated?: boolean;
  total_embedded_positions_count?: number;
  total_changed_positions_count?: number;
  completeness_status?: "COMPLETE" | "PARTIAL" | "UNAVAILABLE" | "FAILED";
  error_message?: string | null;
}

export interface PixelDiffMetrics {
  changed_pixels_count: number;
  max_channel_diff: number;
  total_pixels: number;
  changed_percentage: number;
}

interface StegoInspectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  stegoImageUrl: string | null;
  coverImageUrl: string | null;
  diagnostics: DiagnosticsMetadata | null;
  pixelDiffMetrics: PixelDiffMetrics | null;
  psnr: number | null;
  ssim: number | null;
}

export const StegoInspectorModal: React.FC<StegoInspectorModalProps> = ({
  isOpen,
  onClose,
  stegoImageUrl,
  coverImageUrl,
  diagnostics,
  pixelDiffMetrics,
  psnr,
  ssim,
}) => {
  const [activeView, setActiveView] = useState<InspectorViewMode>("stego");
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [overlayVisible, setOverlayVisible] = useState<boolean>(true);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.75);
  const [amplificationFactor, setAmplificationFactor] = useState<number>(10);

  // Hidden canvases for pixel operations
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Cached ImageData
  const [diffImageUrl, setDiffImageUrl] = useState<string | null>(null);
  const [amplifiedImageUrl, setAmplifiedImageUrl] = useState<string | null>(null);
  const [dimensionError, setDimensionError] = useState<string | null>(null);
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

  // Decode stego image natural dimensions immediately whenever stego image is available
  useEffect(() => {
    if (!isOpen || !stegoImageUrl) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setCarrierDims(prev => {
          if (prev.width !== img.naturalWidth || prev.height !== img.naturalHeight) {
            return { width: img.naturalWidth, height: img.naturalHeight };
          }
          return prev;
        });
      }
    };
    img.src = stegoImageUrl;
  }, [isOpen, stegoImageUrl]);

  const imageWidth = diagnostics?.carrier_dims ? diagnostics.carrier_dims[0] : carrierDims.width;
  const imageHeight = diagnostics?.carrier_dims ? diagnostics.carrier_dims[1] : carrierDims.height;

  const subbandWidth = diagnostics?.subband_width
    || (diagnostics && typeof diagnostics.subband_dim === "object" ? diagnostics.subband_dim[1] : (typeof diagnostics?.subband_dim === "number" ? diagnostics.subband_dim : Math.floor(imageWidth / 2)));
  const subbandHeight = diagnostics?.subband_height
    || (diagnostics && typeof diagnostics.subband_dim === "object" ? diagnostics.subband_dim[0] : (typeof diagnostics?.subband_dim === "number" ? diagnostics.subband_dim : Math.floor(imageHeight / 2)));

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Reset zoom on open
  useEffect(() => {
    if (isOpen) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  }, [isOpen]);

  // Load and decode pixel data from stego and cover images
  useEffect(() => {
    if (!isOpen || !stegoImageUrl || !coverImageUrl) return;

    let isSubscribed = true;
    const coverImg = new Image();
    const stegoImg = new Image();
    let loadedCount = 0;

    const onImageLoaded = () => {
      loadedCount++;
      if (loadedCount === 2 && isSubscribed) {
        const cW = coverImg.naturalWidth;
        const cH = coverImg.naturalHeight;
        const sW = stegoImg.naturalWidth;
        const sH = stegoImg.naturalHeight;

        if (cW <= 0 || cH <= 0 || sW <= 0 || sH <= 0 || cW !== sW || cH !== sH || cW % 2 !== 0 || cH % 2 !== 0) {
          setDimensionError(
            `Dimension Incompatibility: Carrier is ${cW}×${cH}, Stego is ${sW}×${sH}. Direct pixel comparison requires matching even-dimension PNG carriers.`
          );
          setDiffImageUrl(null);
          setAmplifiedImageUrl(null);
          return;
        }

        setCarrierDims({ width: cW, height: cH });
        setDimensionError(null);
        const offscreen = document.createElement("canvas");
        offscreen.width = cW;
        offscreen.height = cH;
        const ctx = offscreen.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;

        // Extract carrier pixels (draw 1:1 without resampling)
        ctx.drawImage(coverImg, 0, 0);
        const cData = ctx.getImageData(0, 0, cW, cH);

        // Extract stego pixels (draw 1:1 without resampling)
        ctx.clearRect(0, 0, cW, cH);
        ctx.drawImage(stegoImg, 0, 0);
        const sData = ctx.getImageData(0, 0, cW, cH);

        // Compute direct grayscale difference
        const directDiff = computePixelDifference(
          cData.data,
          sData.data,
          { width: cW, height: cH, channels: 4 },
          { width: cW, height: cH, channels: 4 },
          { mode: "grayscale" }
        );
        const diffImgData = ctx.createImageData(cW, cH);
        diffImgData.data.set(directDiff.outputPixels);
        ctx.putImageData(diffImgData, 0, 0);
        setDiffImageUrl(offscreen.toDataURL("image/png"));

        // Compute amplified difference
        const ampDiff = computePixelDifference(
          cData.data,
          sData.data,
          { width: cW, height: cH, channels: 4 },
          { width: cW, height: cH, channels: 4 },
          { mode: "amplified", amplificationFactor }
        );
        const ampImgData = ctx.createImageData(cW, cH);
        ampImgData.data.set(ampDiff.outputPixels);
        ctx.putImageData(ampImgData, 0, 0);
        setAmplifiedImageUrl(offscreen.toDataURL("image/png"));
      }
    };

    coverImg.crossOrigin = "anonymous";
    stegoImg.crossOrigin = "anonymous";
    coverImg.onload = onImageLoaded;
    stegoImg.onload = onImageLoaded;
    coverImg.src = coverImageUrl;
    stegoImg.src = stegoImageUrl;

    return () => {
      isSubscribed = false;
    };
  }, [isOpen, stegoImageUrl, coverImageUrl, amplificationFactor]);

  // Draw overlay canvas for embedded locations or changed coefficients
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isOpen) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!overlayVisible) return;

    // Calculate explicit image viewport rectangle within canvas (handles 1:1 as well as letterbox/pillarbox)
    const viewport = calculateImageViewport(
      canvas.width,
      canvas.height,
      imageWidth,
      imageHeight,
      "contain"
    );

    // Enforce image-boundary clipping: no marker can render outside the displayed image rectangle
    ctx.save();
    ctx.beginPath();
    ctx.rect(viewport.offsetX, viewport.offsetY, viewport.displayedWidth, viewport.displayedHeight);
    ctx.clip();

    if (activeView === "embedded" && diagnostics && diagnostics.embedded_positions) {
      ctx.fillStyle = `rgba(239, 68, 68, ${overlayOpacity})`; // Red-500
      for (const pos of diagnostics.embedded_positions) {
        if (!pos || !Array.isArray(pos) || pos.length < 2) continue;
        const [r, c] = pos;
        const marker = mapSubbandToCanvas(r, c, subbandHeight, subbandWidth, viewport);
        if (!marker) continue; // Out of bounds -> strictly rejected
        ctx.fillRect(marker.x, marker.y, marker.width, marker.height);
      }
    } else if (activeView === "changed" && diagnostics && diagnostics.changed_positions) {
      ctx.fillStyle = `rgba(6, 182, 212, ${overlayOpacity})`; // Cyan-500
      for (const pos of diagnostics.changed_positions) {
        if (!pos || !Array.isArray(pos) || pos.length < 2) continue;
        const [r, c] = pos;
        const marker = mapSubbandToCanvas(r, c, subbandHeight, subbandWidth, viewport);
        if (!marker) continue; // Out of bounds -> strictly rejected
        ctx.fillRect(marker.x, marker.y, marker.width, marker.height);
      }
    }

    ctx.restore();
  }, [
    activeView,
    diagnostics,
    overlayVisible,
    overlayOpacity,
    isOpen,
    imageWidth,
    imageHeight,
    subbandWidth,
    subbandHeight
  ]);

  if (!isOpen) return null;

  // Zoom handlers
  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.3, 10));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.3, 0.5));
  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom(prev => Math.min(prev * 1.15, 10));
    } else {
      setZoom(prev => Math.max(prev / 1.15, 0.5));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleDownloadPristine = () => {
    if (!stegoImageUrl) return;
    const a = document.createElement("a");
    a.href = stegoImageUrl;
    a.download = `stego_carrier_${diagnostics?.run_id || "output"}.png`;
    a.click();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 overflow-hidden">
      <div
        className="bg-[#121212] border border-[#2b2b2b] rounded-xl flex flex-col w-full max-w-6xl max-h-[92vh] shadow-2xl overflow-hidden font-sans select-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* HEADER */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262626] bg-[#161616]">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded bg-primary/20 border border-primary/40 flex items-center justify-center">
              <Layers className="text-primary w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold font-mono tracking-wide text-[#ededed]">
                  STEGO-MEDIA EMBEDDING INSPECTOR
                </h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-primary/20 text-primary border border-primary/30">
                  EVIDENCE-DRIVEN
                </span>
              </div>
              <p className="text-xs text-muted font-mono">
                Inspect authentic transform-domain embedding locations and spatial carrier modifications
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={handleDownloadPristine}
              className="text-xs font-mono bg-[#1f1f1f] hover:bg-[#2a2a2a] border border-[#333] text-[#ededed] px-3 py-1.5 rounded flex items-center transition-colors"
              title="Download untouched 8-bit RGB Stego PNG"
            >
              <Download className="w-3.5 h-3.5 mr-1.5 text-primary" />
              Download Pristine Stego PNG
            </button>
            <button
              onClick={onClose}
              className="text-muted hover:text-[#ededed] p-1.5 rounded-lg hover:bg-[#222] transition-colors"
              aria-label="Close Inspector"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* METRICS & RUN SUMMARY STRIP */}
        <div className="bg-[#0c0c0c] px-6 py-2.5 border-b border-[#222] flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center space-x-6">
            <span className="text-muted">
              Carrier Dimensions: <strong className="text-[#ededed]">
                {diagnostics?.carrier_dims ? `${diagnostics.carrier_dims[0]}×${diagnostics.carrier_dims[1]} RGB` : `${carrierDims.width}×${carrierDims.height} RGB`}
              </strong>
            </span>
            <span className="text-muted">
              PSNR: <strong className="text-green-400">{psnr !== null ? `${psnr} dB` : "N/A"}</strong>
            </span>
            <span className="text-muted">
              SSIM: <strong className="text-green-400">{ssim !== null ? ssim : "N/A"}</strong>
            </span>
            <span className="text-muted flex items-center">
              Verification:{" "}
              <strong className={`ml-1 px-1.5 py-0.5 rounded text-[10px] ${
                diagnostics?.verification_status === "VERIFIED"
                  ? "bg-green-500/20 text-green-400 border border-green-500/30"
                  : diagnostics?.verification_status === "PENDING"
                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                  : "bg-red-500/20 text-red-400 border border-red-500/30"
              }`}>
                {diagnostics?.verification_status || "UNKNOWN"}
              </strong>
            </span>
          </div>

          <div className="flex items-center space-x-6">
            {(diagnostics?.coordinates_truncated || diagnostics?.completeness_status === "PARTIAL") && (
              <span
                className="text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded text-[10px] font-bold"
                title="Diagnostic coordinates were bounded to prevent memory overflow; truthful total counts preserved."
              >
                PARTIAL DIAGNOSTICS (Displaying {diagnostics.embedded_positions?.length || 0} of {(diagnostics.total_embedded_positions_count ?? diagnostics.total_embedded_bits).toLocaleString()} assigned, {diagnostics.changed_positions?.length || 0} of {(diagnostics.total_changed_positions_count ?? diagnostics.changed_coefficients_count).toLocaleString()} changed)
              </span>
            )}
            <span className="text-muted">
              Changed Pixels: <strong className="text-amber-400">
                {pixelDiffMetrics ? `${pixelDiffMetrics.changed_pixels_count.toLocaleString()} (${pixelDiffMetrics.changed_percentage}%)` : "N/A"}
              </strong>
            </span>
            <span className="text-muted">
              Max Channel Diff: <strong className="text-amber-400">
                {pixelDiffMetrics ? `±${pixelDiffMetrics.max_channel_diff} / 255` : "N/A"}
              </strong>
            </span>
            <span className="text-muted">
              Embedded Positions: <strong className="text-primary">
                {diagnostics ? diagnostics.total_embedded_bits.toLocaleString() : "N/A"}
              </strong>
            </span>
            <span className="text-muted">
              Changed HH: <strong className="text-cyan-400">
                {diagnostics ? diagnostics.changed_coefficients_count.toLocaleString() : "N/A"}
              </strong>
            </span>
          </div>
        </div>

        {/* VIEW SELECTOR & INSPECTION CONTROLS */}
        <div className="bg-[#141414] px-6 py-3 border-b border-[#262626] flex flex-wrap items-center justify-between gap-4">

          {/* 5 VIEW TABS */}
          <div className="flex items-center bg-[#0a0a0a] p-1 rounded-lg border border-[#2a2a2a] text-xs font-mono">
            <button
              onClick={() => setActiveView("stego")}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeView === "stego"
                  ? "bg-primary text-white font-semibold"
                  : "text-muted hover:text-[#ededed]"
              }`}
            >
              1. Stego Image
            </button>
            <button
              onClick={() => setActiveView("embedded")}
              className={`px-3 py-1.5 rounded transition-colors flex items-center ${
                activeView === "embedded"
                  ? "bg-red-500 text-white font-semibold"
                  : "text-muted hover:text-[#ededed]"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-red-400 mr-1.5"></span>
              2. Embedding Locations
            </button>
            <button
              onClick={() => setActiveView("changed")}
              className={`px-3 py-1.5 rounded transition-colors flex items-center ${
                activeView === "changed"
                  ? "bg-cyan-600 text-white font-semibold"
                  : "text-muted hover:text-[#ededed]"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-cyan-400 mr-1.5"></span>
              3. Changed Coefficients
            </button>
            <button
              onClick={() => setActiveView("diff")}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeView === "diff"
                  ? "bg-[#333] text-white font-semibold"
                  : "text-muted hover:text-[#ededed]"
              }`}
            >
              4. Pixel Difference
            </button>
            <button
              onClick={() => setActiveView("amplified")}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeView === "amplified"
                  ? "bg-amber-600 text-white font-semibold"
                  : "text-muted hover:text-[#ededed]"
              }`}
            >
              5. Amplified Difference ({amplificationFactor}×)
            </button>
          </div>

          {/* ZOOM & OVERLAY CONTROLS */}
          <div className="flex items-center space-x-3 text-xs font-mono">
            {/* Overlay toggle */}
            {(activeView === "embedded" || activeView === "changed") && (
              <div className="flex items-center space-x-2 mr-2 bg-[#0d0d0d] px-2.5 py-1 rounded border border-[#2a2a2a]">
                <button
                  onClick={() => setOverlayVisible(!overlayVisible)}
                  className="flex items-center text-muted hover:text-[#ededed]"
                >
                  {overlayVisible ? <Eye className="w-3.5 h-3.5 mr-1 text-primary" /> : <EyeOff className="w-3.5 h-3.5 mr-1 text-muted" />}
                  <span>Overlay {overlayVisible ? "ON" : "OFF"}</span>
                </button>
                <div className="w-px h-3 bg-[#333]"></div>
                <div className="flex items-center space-x-1">
                  <Sliders className="w-3 h-3 text-muted" />
                  <input
                    type="range"
                    min="0.2"
                    max="1.0"
                    step="0.05"
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                    className="w-16 accent-primary h-1 bg-[#222] rounded cursor-pointer"
                  />
                  <span className="text-[10px] text-muted w-7 text-right">
                    {Math.round(overlayOpacity * 100)}%
                  </span>
                </div>
              </div>
            )}

            {/* Amplification factor selector */}
            {activeView === "amplified" && (
              <div className="flex items-center space-x-1 mr-2 bg-[#0d0d0d] px-2 py-1 rounded border border-[#2a2a2a]">
                <span className="text-muted text-[10px]">Amp:</span>
                {[5, 10, 20, 50].map((factor) => (
                  <button
                    key={factor}
                    onClick={() => setAmplificationFactor(factor)}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      amplificationFactor === factor
                        ? "bg-amber-600 text-white"
                        : "text-muted hover:text-[#ededed]"
                    }`}
                  >
                    {factor}×
                  </button>
                ))}
              </div>
            )}

            {/* Zoom toolbar */}
            <div className="flex items-center bg-[#0d0d0d] rounded border border-[#2a2a2a] p-1 space-x-1">
              <button
                onClick={handleZoomOut}
                className="p-1 hover:bg-[#222] rounded text-muted hover:text-[#ededed]"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="px-1 text-[11px] text-muted min-w-[3.5rem] text-center">
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={handleZoomIn}
                className="p-1 hover:bg-[#222] rounded text-muted hover:text-[#ededed]"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={handleResetZoom}
                className="p-1 hover:bg-[#222] rounded text-muted hover:text-[#ededed] ml-1"
                title="Reset Zoom"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* MAIN VIEWER CANVAS CONTAINER */}
        <div
          ref={containerRef}
          className="flex-1 bg-[#050505] overflow-hidden relative flex items-center justify-center p-6 cursor-grab active:cursor-grabbing"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {stegoImageUrl ? (
            <div
              className="relative transition-transform duration-75 select-none"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
                width: imageWidth,
                height: imageHeight,
              }}
            >
              {/* Layer 1: Base Image (Stego, Diff, or Amplified) */}
              {activeView === "diff" ? (
                dimensionError ? (
                  <div
                    style={{ width: imageWidth, height: imageHeight }}
                    className="flex flex-col items-center justify-center p-6 text-center bg-[#111] border border-amber-500/30 rounded font-mono"
                  >
                    <Info className="w-8 h-8 text-amber-400 mb-2" />
                    <span className="text-xs font-bold text-amber-300 mb-1">PIXEL COMPARISON INCOMPATIBLE</span>
                    <span className="text-[11px] text-muted max-w-sm">{dimensionError}</span>
                  </div>
                ) : diffImageUrl ? (
                  <img
                    src={diffImageUrl}
                    alt="Pixel Difference Map"
                    style={{ width: imageWidth, height: imageHeight }}
                    className="w-full h-full block pointer-events-none image-pixelated shadow-lg max-w-none max-h-none"
                  />
                ) : (
                  <div
                    style={{ width: imageWidth, height: imageHeight }}
                    className="flex items-center justify-center text-muted font-mono text-xs"
                  >
                    Computing pixel difference...
                  </div>
                )
              ) : activeView === "amplified" ? (
                dimensionError ? (
                  <div
                    style={{ width: imageWidth, height: imageHeight }}
                    className="flex flex-col items-center justify-center p-6 text-center bg-[#111] border border-amber-500/30 rounded font-mono"
                  >
                    <Info className="w-8 h-8 text-amber-400 mb-2" />
                    <span className="text-xs font-bold text-amber-300 mb-1">AMPLIFIED COMPARISON INCOMPATIBLE</span>
                    <span className="text-[11px] text-muted max-w-sm">{dimensionError}</span>
                  </div>
                ) : amplifiedImageUrl ? (
                  <img
                    src={amplifiedImageUrl}
                    alt="Amplified Difference Map"
                    style={{ width: imageWidth, height: imageHeight }}
                    className="w-full h-full block pointer-events-none image-pixelated shadow-lg max-w-none max-h-none"
                  />
                ) : (
                  <div
                    style={{ width: imageWidth, height: imageHeight }}
                    className="flex items-center justify-center text-muted font-mono text-xs"
                  >
                    Computing amplified difference...
                  </div>
                )
              ) : (
                <img
                  src={stegoImageUrl}
                  alt="Stego Container"
                  style={{ width: imageWidth, height: imageHeight }}
                  className="w-full h-full block pointer-events-none image-pixelated shadow-lg max-w-none max-h-none"
                />
              )}

              {/* Layer 2: Canvas Overlay for Embedding / Changed Positions */}
              {(activeView === "embedded" || activeView === "changed") && (
                <canvas
                  ref={canvasRef}
                  width={imageWidth}
                  height={imageHeight}
                  style={{ width: imageWidth, height: imageHeight }}
                  className="absolute inset-0 pointer-events-none block max-w-none max-h-none"
                />
              )}
            </div>
          ) : (
            <div className="text-center font-mono text-muted text-xs">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-40 text-muted" />
              <span>No stego image available. Execute a pipeline run first.</span>
            </div>
          )}

          {/* Floater Legend & Transform Domain Support Label */}
          <div className="absolute bottom-4 left-6 bg-[#121212]/90 border border-[#2b2b2b] px-3.5 py-2.5 rounded-lg text-[11px] font-mono text-muted backdrop-blur-md max-w-md shadow-lg">
            <div className="flex items-center text-primary font-bold mb-1">
              <Info className="w-3.5 h-3.5 mr-1.5 flex-shrink-0" />
              <span>
                {activeView === "stego" && "Untouched Lossless Stego Container (8-bit RGB PNG)"}
                {activeView === "embedded" && "Assigned Embedding Positions (Red Markers)"}
                {activeView === "changed" && "Numerically Changed Coefficients (Cyan Markers)"}
                {activeView === "diff" && "Absolute Pixel Differences (|Stego - Carrier|)"}
                {activeView === "amplified" && `Amplified Pixel Differences (${amplificationFactor}× Multiplier)`}
              </span>
            </div>
            <p className="text-[10px] leading-relaxed">
              {activeView === "stego" && "Pristine carrier reconstructed from inverse DWT. Zero markers drawn into pixels; bit-exact to backend execution."}
              {activeView === "embedded" && "Each marker highlights the 2×2 carrier pixel block corresponding to a modulated Red-HH subband coefficient."}
              {activeView === "changed" && "Highlights subband coefficients where Modified HH' differs from Original HH (|Δ| > 1e-4)."}
              {activeView === "diff" && "Direct channel difference map. High fidelity stego results in near-black preview (max diff ≤ 15 levels)."}
              {activeView === "amplified" && `Differences scaled by ${amplificationFactor}× for visual inspection only. Does not alter the underlying media container.`}
            </p>
          </div>
        </div>

        {/* FOOTER */}
        <div className="px-6 py-3 border-t border-[#262626] bg-[#161616] flex items-center justify-between text-xs font-mono text-muted">
          <span>
            Run ID: <strong className="text-[#ededed]">{diagnostics?.run_id || "N/A"}</strong>
          </span>
          <div className="flex items-center space-x-4">
            <span>Scroll wheel to zoom | Drag to pan</span>
            <button
              onClick={onClose}
              className="bg-[#222] hover:bg-[#2b2b2b] border border-[#333] text-[#ededed] px-4 py-1.5 rounded transition-colors"
            >
              Close Inspector (Esc)
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
