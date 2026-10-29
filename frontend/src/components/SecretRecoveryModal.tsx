import React, { useState, useEffect } from 'react';
import { X, CheckCircle2, XCircle, ZoomIn, ShieldCheck, Download } from 'lucide-react';

interface SecretRecoveryModalProps {
  isOpen: boolean;
  onClose: () => void;
  originalImageUrl: string | null;
  recoveredImageUrl: string | null;
  originalFilename: string;
  originalBytes: number;
  recoveredBytes: number;
  originalDims?: [number, number] | null;
  recoveredDims?: [number, number] | null;
  exactMatch: boolean | null;
}

export const SecretRecoveryModal: React.FC<SecretRecoveryModalProps> = ({
  isOpen,
  onClose,
  originalImageUrl,
  recoveredImageUrl,
  originalFilename,
  originalBytes,
  recoveredBytes,
  originalDims,
  recoveredDims,
  exactMatch,
}) => {
  const [zoomMultiplier, setZoomMultiplier] = useState<number>(8); // 8x default for small icons

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

  if (!isOpen) return null;

  const handleDownloadRecovered = () => {
    if (!recoveredImageUrl) return;
    const a = document.createElement("a");
    a.href = recoveredImageUrl;
    a.download = `recovered_${originalFilename || "secret.png"}`;
    a.click();
  };

  const dimsText = (dims?: [number, number] | null) => {
    if (!dims || dims[0] <= 0 || dims[1] <= 0) return "N/A";
    return `${dims[0]}×${dims[1]}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 overflow-hidden">
      <div
        className="bg-[#121212] border border-[#2b2b2b] rounded-xl flex flex-col w-full max-w-4xl max-h-[90vh] shadow-2xl overflow-hidden font-sans select-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* HEADER */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262626] bg-[#161616]">
          <div className="flex items-center space-x-3">
            <div className={`w-8 h-8 rounded border flex items-center justify-center ${
              exactMatch
                ? "bg-green-500/20 border-green-500/40 text-green-400"
                : "bg-red-500/20 border-red-500/40 text-red-400"
            }`}>
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold font-mono tracking-wide text-[#ededed]">
                  SECRET MEDIA RECOVERY VERIFICATION
                </h2>
                {exactMatch !== null && (
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold flex items-center ${
                    exactMatch
                      ? "bg-green-500/20 text-green-400 border border-green-500/30"
                      : "bg-red-500/20 text-red-400 border border-red-500/30"
                  }`}>
                    {exactMatch ? <CheckCircle2 className="w-3 h-3 mr-1" /> : <XCircle className="w-3 h-3 mr-1" />}
                    {exactMatch ? "EXACT MATCH VERIFIED" : "VERIFICATION FAILED"}
                  </span>
                )}
              </div>
              <p className="text-xs text-muted font-mono">
                Cryptographic verbatim byte-for-byte equality verification against original input media
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="text-muted hover:text-[#ededed] p-1.5 rounded-lg hover:bg-[#222] transition-colors"
            aria-label="Close Modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* VERIFICATION BANNER */}
        <div className={`px-6 py-3 border-b flex items-center justify-between text-xs font-mono ${
          exactMatch
            ? "bg-green-950/20 border-green-500/30 text-green-400"
            : "bg-red-950/20 border-red-500/30 text-red-400"
        }`}>
          <div className="flex items-center space-x-2">
            {exactMatch ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
            <span className="font-semibold">
              {exactMatch
                ? "100% BYTE-FOR-BYTE EQUALITY CONFIRMED: AES-256-GCM authenticated decryption & CRC-32 integrity passed."
                : "MISMATCH: Recovered payload does not match original secret input verbatim."}
            </span>
          </div>

          {/* ZOOM SELECTOR */}
          <div className="flex items-center space-x-1.5 text-muted">
            <ZoomIn className="w-3.5 h-3.5 mr-1" />
            <span className="text-[10px]">Magnification:</span>
            {[1, 2, 4, 8, 16].map((z) => (
              <button
                key={z}
                onClick={() => setZoomMultiplier(z)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  zoomMultiplier === z ? "bg-[#333] text-white" : "hover:text-[#ededed]"
                }`}
              >
                {z}×
              </button>
            ))}
          </div>
        </div>

        {/* SIDE-BY-SIDE COMPARISON */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-[#0a0a0a] flex-1 overflow-y-auto">

          {/* ORIGINAL SECRET CARD */}
          <div className="border border-[#262626] rounded-xl p-4 bg-[#111111] flex flex-col">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-[#222] text-xs font-mono">
              <span className="font-bold text-primary">ORIGINAL INPUT SECRET MEDIA</span>
              <span className="text-muted text-[10px] bg-[#1a1a1a] px-2 py-0.5 rounded">
                {dimsText(originalDims)} | {originalBytes} B
              </span>
            </div>

            <div className="flex-1 min-h-[280px] bg-[#070707] rounded border border-[#1f1f1f] flex items-center justify-center p-4 overflow-auto">
              {originalImageUrl ? (
                <img
                  src={originalImageUrl}
                  alt="Original Secret"
                  className="image-pixelated object-contain border border-[#333] shadow-md"
                  style={{
                    transform: `scale(${zoomMultiplier})`,
                    transformOrigin: "center center",
                  }}
                />
              ) : (
                <span className="text-xs font-mono text-muted">No original image loaded</span>
              )}
            </div>

            <div className="mt-3 text-[11px] font-mono text-muted flex justify-between">
              <span>File: <strong className="text-[#ededed]">{originalFilename}</strong></span>
              <span>Bytes: <strong className="text-[#ededed]">{originalBytes}</strong></span>
            </div>
          </div>

          {/* RECOVERED SECRET CARD */}
          <div className="border border-[#262626] rounded-xl p-4 bg-[#111111] flex flex-col">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-[#222] text-xs font-mono">
              <span className="font-bold text-cyan-400">RECONSTRUCTED RECOVERED MEDIA</span>
              <span className="text-muted text-[10px] bg-[#1a1a1a] px-2 py-0.5 rounded">
                {dimsText(recoveredDims)} | {recoveredBytes} B
              </span>
            </div>

            <div className="flex-1 min-h-[280px] bg-[#070707] rounded border border-[#1f1f1f] flex items-center justify-center p-4 overflow-auto">
              {recoveredImageUrl ? (
                <img
                  src={recoveredImageUrl}
                  alt="Recovered Secret"
                  className="image-pixelated object-contain border border-[#333] shadow-md"
                  style={{
                    transform: `scale(${zoomMultiplier})`,
                    transformOrigin: "center center",
                  }}
                />
              ) : (
                <span className="text-xs font-mono text-muted">Awaiting receiver recovery</span>
              )}
            </div>

            <div className="mt-3 text-[11px] font-mono text-muted flex justify-between items-center">
              <span>Bytes: <strong className="text-[#ededed]">{recoveredBytes}</strong></span>
              <button
                onClick={handleDownloadRecovered}
                disabled={!recoveredImageUrl}
                className="text-[11px] bg-[#1c1c1c] hover:bg-[#262626] border border-[#333] text-[#ededed] px-2.5 py-1 rounded flex items-center disabled:opacity-30"
              >
                <Download className="w-3 h-3 mr-1 text-primary" />
                Export Recovered Media
              </button>
            </div>
          </div>

        </div>

        {/* FOOTER */}
        <div className="px-6 py-3 border-t border-[#262626] bg-[#161616] flex items-center justify-between text-xs font-mono text-muted">
          <span>
            Verification Protocol: <strong className="text-[#ededed]">SHA-256 + CRC-32 + Byte-for-Byte Exact Equality</strong>
          </span>
          <button
            onClick={onClose}
            className="bg-[#222] hover:bg-[#2b2b2b] border border-[#333] text-[#ededed] px-4 py-1.5 rounded transition-colors"
          >
            Close Window (Esc)
          </button>
        </div>

      </div>
    </div>
  );
};
