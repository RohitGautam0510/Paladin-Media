import React, { useState, useEffect, useRef } from 'react';
import type { PipelineEvent, PipelineStage, StageStatus, DefaultFixtures } from './services/backendAdapter';
import { backend } from './services/backendAdapter';
import {
  Shield,
  Play,
  Square,
  Activity,
  Upload,
  Image as ImageIcon,
  CheckCircle2,
  XCircle,
  Lock,
  Zap,
  Eye,
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { StegoInspectorModal, type DiagnosticsMetadata, type PixelDiffMetrics } from './components/StegoInspectorModal';
import { DwtSubbandInspector } from './components/DwtSubbandInspector';
import { SecretRecoveryModal } from './components/SecretRecoveryModal';

type StageState = {
  status: StageStatus;
  message?: string;
  metadata?: any;
};

const STAGES: PipelineStage[] = [
  "LOAD",
  "SERIALIZE",
  "ENCRYPT",
  "RS_ENCODE",
  "CAPACITY_CHECK",
  "DWT",
  "EMBED",
  "IDWT",
  "RECEIVER_DWT",
  "EXTRACT",
  "RS_DECODE",
  "AES_VERIFY",
  "SECRET_RECOVER",
  "EXACT_MATCH_VERIFY",
];

const STAGE_LABELS: Record<PipelineStage, string> = {
  "LOAD": "Load Media",
  "SERIALIZE": "Payload Serialization (CRC-32)",
  "ENCRYPT": "AES-256-GCM Encryption",
  "RS_ENCODE": "Reed-Solomon ECC Encoding",
  "CAPACITY_CHECK": "Carrier Capacity Hard Gate",
  "DWT": "1-Level 2D Haar DWT",
  "EMBED": "Deterministic Parity Embedding",
  "IDWT": "Inverse DWT & PNG Synthesis",
  "RECEIVER_DWT": "Receiver Stego DWT",
  "EXTRACT": "Header & Payload Extraction",
  "RS_DECODE": "Reed-Solomon Error Correction",
  "AES_VERIFY": "AES-GCM Tag Verification",
  "SECRET_RECOVER": "Secret Media Reconstruction",
  "EXACT_MATCH_VERIFY": "Exact Byte Equality Verification",
  "ERROR": "Pipeline Error",
};

export type RunLifecycleStatus = "IDLE" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";

export default function App() {
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<RunLifecycleStatus>("IDLE");
  const [stages, setStages] = useState<Record<PipelineStage, StageState>>(
    STAGES.reduce((acc, s) => ({ ...acc, [s]: { status: "waiting" } }), {} as any)
  );
  const [logs, setLogs] = useState<string[]>([]);

  // Media state
  const [secretSessionId, setSecretSessionId] = useState<string | null>(null);
  const [secretPreview, setSecretPreview] = useState<string | null>(null);
  const [secretFilename, setSecretFilename] = useState<string>("No file chosen");
  const [secretBytesSize, setSecretBytesSize] = useState<number>(0);
  const [secretDims, setSecretDims] = useState<{ width: number; height: number; format?: string } | null>(null);

  const [coverSessionId, setCoverSessionId] = useState<string | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const [coverFilename, setCoverFilename] = useState<string>("No file chosen");
  const [coverDims, setCoverDims] = useState<{ width: number; height: number; format?: string; mode?: string } | null>(null);
  const [isDatasetCustom, setIsDatasetCustom] = useState<boolean>(false);

  // Dynamic capacity limits
  const [capacityLimits, setCapacityLimits] = useState<{
    raw_secret_capacity_bytes: number;
    max_protected_bytes: number;
    subband_width: number;
    subband_height: number;
  }>({
    raw_secret_capacity_bytes: 8192,
    max_protected_bytes: 8176,
    subband_width: 256,
    subband_height: 256,
  });

  // Output artifacts
  const [stegoImage, setStegoImage] = useState<string | null>(null);
  const [recoveredImage, setRecoveredImage] = useState<string | null>(null);
  const [dwtSubbands, setDwtSubbands] = useState<Record<string, string> | null>(null);
  const [modifiedHhImage, setModifiedHhImage] = useState<string | null>(null);
  const [psnr, setPsnr] = useState<number | null>(null);
  const [ssim, setSsim] = useState<number | null>(null);
  const [exactMatch, setExactMatch] = useState<boolean | null>(null);

  // Diagnostic metadata & Run isolation
  const [diagnostics, setDiagnostics] = useState<DiagnosticsMetadata | null>(null);
  const [pixelDiffMetrics, setPixelDiffMetrics] = useState<PixelDiffMetrics | null>(null);
  const [recoveredBytes, setRecoveredBytes] = useState<number>(0);
  const [recoveredDims, setRecoveredDims] = useState<[number, number] | null>(null);
  const [originalDims, setOriginalDims] = useState<[number, number] | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  // Inspection Modals
  const [isStegoInspectorOpen, setIsStegoInspectorOpen] = useState<boolean>(false);
  const [isSecretRecoveryOpen, setIsSecretRecoveryOpen] = useState<boolean>(false);

  // Capacity metadata
  const [capacityInfo, setCapacityInfo] = useState<any>(null);

  const logRef = useRef<HTMLDivElement>(null);

  const loadBaselineFixtures = () => {
    backend.fetchDefaultFixtures()
      .then((fixtures: DefaultFixtures) => {
        if (fixtures.secret && fixtures.secret.data_url) {
          setSecretSessionId(fixtures.secret.session_id);
          setSecretPreview(fixtures.secret.data_url);
          setSecretFilename(fixtures.secret.name);
          setSecretBytesSize(fixtures.secret.size);
          setSecretDims({
            width: fixtures.secret.width || 24,
            height: fixtures.secret.height || 24,
            format: fixtures.secret.format || "PNG",
          });
        }
        if (fixtures.cover && fixtures.cover.data_url) {
          setCoverSessionId(fixtures.cover.session_id);
          setCoverPreview(fixtures.cover.data_url);
          setCoverFilename(fixtures.cover.name);
          const cW = fixtures.cover.width || 512;
          const cH = fixtures.cover.height || 512;
          setCoverDims({
            width: cW,
            height: cH,
            format: fixtures.cover.format || "PNG",
            mode: fixtures.cover.mode || "RGB",
          });
          setIsDatasetCustom(false);
          backend.fetchCarrierCapacity(cW, cH)
            .then(cap => setCapacityLimits(cap))
            .catch(() => {});
        }
        addLog("Preloaded baseline fixtures: secret_small.png (24x24) & 0001x2.png (512x512)");
      })
      .catch((err) => {
        console.warn("Could not load default fixtures automatically:", err);
      });
  };

  useEffect(() => {
    loadBaselineFixtures();
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, `[${timestamp}] ${msg}`]);
  };

  const handleSecretUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSecretFilename(file.name);
      setSecretBytesSize(file.size);
      const url = URL.createObjectURL(file);
      setSecretPreview(url);
      addLog(`Selected secret media file: ${file.name} (${file.size} bytes)`);

      const sImg = new Image();
      sImg.onload = () => {
        setSecretDims({ width: sImg.naturalWidth, height: sImg.naturalHeight });
        addLog(`Secret image dimensions: ${sImg.naturalWidth}×${sImg.naturalHeight}`);
      };
      sImg.src = url;

      try {
        const uploadRes = await backend.uploadMedia(file);
        setSecretSessionId(uploadRes.session_id);
        if (uploadRes.width && uploadRes.height) {
          setSecretDims({ width: uploadRes.width, height: uploadRes.height, format: uploadRes.format });
        }
        addLog(`Secret file uploaded to session ${uploadRes.session_id}`);
      } catch (err: any) {
        addLog(`ERROR: Failed to upload secret file: ${err.message}`);
      }
    }
  };

  const handleCoverUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setCoverFilename(file.name);
      const url = URL.createObjectURL(file);
      setCoverPreview(url);
      addLog(`Selected cover carrier image: ${file.name} (${file.size} bytes)`);

      const cImg = new Image();
      cImg.onload = () => {
        const w = cImg.naturalWidth;
        const h = cImg.naturalHeight;
        setCoverDims({ width: w, height: h, format: "PNG", mode: "RGB" });
        setIsDatasetCustom(true);
        addLog(`Cover image dimensions: ${w}×${h} RGB`);

        if (w < 32 || h < 32) {
          addLog(`WARNING: Carrier dimensions ${w}×${h} are below minimum 32×32.`);
        } else if (w % 2 !== 0 || h % 2 !== 0) {
          addLog(`WARNING: Carrier dimensions ${w}×${h} are odd. DWT decomposition requires even dimensions.`);
        } else {
          backend.fetchCarrierCapacity(w, h)
            .then(cap => {
              setCapacityLimits(cap);
              addLog(`Carrier capacity: ${cap.raw_secret_capacity_bytes.toLocaleString()} B raw, ${cap.max_protected_bytes.toLocaleString()} B max protected.`);
            })
            .catch(err => {
              addLog(`Capacity calculation error: ${err.message}`);
            });
        }
      };
      cImg.src = url;

      try {
        const uploadRes = await backend.uploadMedia(file);
        setCoverSessionId(uploadRes.session_id);
        if (uploadRes.width && uploadRes.height) {
          setCoverDims({ width: uploadRes.width, height: uploadRes.height, format: uploadRes.format, mode: uploadRes.mode });
        }
        addLog(`Cover file uploaded to session ${uploadRes.session_id}`);
      } catch (err: any) {
        addLog(`ERROR: Failed to upload cover file: ${err.message}`);
      }
    }
  };

  const resetOutputs = () => {
    setStegoImage(null);
    setRecoveredImage(null);
    setDwtSubbands(null);
    setModifiedHhImage(null);
    setPsnr(null);
    setSsim(null);
    setExactMatch(null);
    setCapacityInfo(null);
    setDiagnostics(null);
    setPixelDiffMetrics(null);
    setRecoveredBytes(0);
    setRecoveredDims(null);
    setOriginalDims(null);
    setActiveRunId(null);
    setIsStegoInspectorOpen(false);
    setIsSecretRecoveryOpen(false);
    setStages(STAGES.reduce((acc, s) => ({ ...acc, [s]: { status: "waiting" } }), {} as any));
  };

  const handleRun = () => {
    if (!secretSessionId) {
      addLog("ERROR: Please select or upload secret media first.");
      return;
    }

    setRunning(true);
    setRunStatus("RUNNING");
    resetOutputs();
    addLog("--- RUN INITIALIZED: Post-Quantum Media Steganography Pipeline ---");

    backend.runMediaExperiment(
      secretSessionId,
      coverSessionId,
      (event: PipelineEvent) => {
        setStages(prev => ({
          ...prev,
          [event.stage]: {
            status: event.status,
            message: event.message,
            metadata: { ...prev[event.stage]?.metadata, ...event.metadata },
          }
        }));

        if (event.message) {
          addLog(event.message);
        }

        // Process artifacts from stage metadata
        if (event.metadata?.run_id) {
          setActiveRunId(event.metadata.run_id);
        }
        if (event.stage === "CAPACITY_CHECK" && event.metadata) {
          setCapacityInfo(event.metadata);
        }
        if (event.stage === "DWT" && event.metadata) {
          if (event.metadata.subbands) {
            setDwtSubbands(event.metadata.subbands);
          } else if (event.metadata.run_id) {
            backend.fetchRunSubbands(event.metadata.run_id)
              .then(subbands => setDwtSubbands(subbands))
              .catch(() => {});
          }
        }
        if (event.stage === "EMBED" && event.metadata) {
          if (event.metadata.modified_hh) setModifiedHhImage(event.metadata.modified_hh);
          if (event.metadata.diagnostics) setDiagnostics(event.metadata.diagnostics);
        }
        if (event.stage === "IDWT" && event.metadata) {
          if (event.metadata.stego_image) {
            setStegoImage(event.metadata.stego_image);
          }
          if (event.metadata.stego_session_id) {
            backend.fetchSessionMedia(event.metadata.stego_session_id)
              .then(url => setStegoImage(url))
              .catch(() => {});
          }
          if (event.metadata.psnr_db !== undefined) setPsnr(event.metadata.psnr_db);
          if (event.metadata.ssim !== undefined) setSsim(event.metadata.ssim);
          if (event.metadata.pixel_diff_metrics) setPixelDiffMetrics(event.metadata.pixel_diff_metrics);
        }
        if (event.stage === "SECRET_RECOVER" && event.metadata) {
          if (event.metadata.recovered_image) setRecoveredImage(event.metadata.recovered_image);
          if (event.metadata.recovered_bytes) setRecoveredBytes(event.metadata.recovered_bytes);
          if (event.metadata.recovered_dims) setRecoveredDims(event.metadata.recovered_dims);
          if (event.metadata.original_dims) setOriginalDims(event.metadata.original_dims);
        }
        if (event.stage === "EXACT_MATCH_VERIFY" && event.metadata) {
          const matched = Boolean(event.metadata.exact_match);
          setExactMatch(matched);
          setDiagnostics(prev => prev ? {
            ...prev,
            verification_status: matched ? "VERIFIED" : "FAILED",
            exact_match: matched,
          } : null);
          if (matched) {
            setRunStatus("COMPLETED");
          }
        }

        if (event.status === "failed") {
          setRunning(false);
          setRunStatus("FAILED");
          setExactMatch(false);
          addLog(`PIPELINE TERMINATED: ${event.error || "Stage execution failed"}`);
          setDiagnostics(prev => prev ? {
            ...prev,
            verification_status: "FAILED",
            exact_match: false,
          } : null);
          // Invalidate stale outputs on failure to enforce run isolation
          setStegoImage(null);
          setRecoveredImage(null);
          setPsnr(null);
          setSsim(null);
          setModifiedHhImage(null);
        }
      },
      () => {
        setRunning(false);
        setStages(prev => {
          const isVerified = prev["EXACT_MATCH_VERIFY"]?.status === "complete";
          const isFailed = Object.values(prev).some(s => s.status === "failed");
          if (isVerified) {
            setRunStatus("COMPLETED");
          } else if (isFailed) {
            setRunStatus("FAILED");
          } else {
            setRunStatus(r => (r === "FAILED" || r === "CANCELLED" ? r : "COMPLETED"));
          }
          return prev;
        });
        addLog("Execution cycle concluded.");
      }
    );
  };

  const handleCancel = () => {
    backend.cancel();
    setRunning(false);
    setRunStatus("CANCELLED");
    resetOutputs();
    addLog("Run cancelled by operator. Overlays and state cleared.");
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#ededed] flex flex-col font-sans select-none">

      {/* HEADER */}
      <header className="border-b border-[#262626] bg-[#121212] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded bg-primary/20 border border-primary/40 flex items-center justify-center">
            <Shield className="text-primary w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-wider font-mono">PALADINMEDIA</h1>
              <span className="text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded font-mono font-semibold">
                PHASE 2
              </span>
            </div>
            <p className="text-xs text-muted font-mono">SIGNAL PROCESSING + CARRIER INFRASTRUCTURE</p>
          </div>
        </div>

        <div className="flex items-center space-x-8 text-xs font-mono">
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Carrier Source</span>
            <span className="text-[#ededed] font-medium">
              {isDatasetCustom
                ? `Custom Carrier (${coverDims ? `${coverDims.width}×${coverDims.height}` : "Dynamic"})`
                : `Baseline Fixture (0001x2.png, 512×512)`}
            </span>
          </div>
          {activeRunId && (
            <div className="flex flex-col">
              <span className="text-muted text-[10px] uppercase">Active Run</span>
              <span className="text-primary font-medium truncate max-w-[120px]">{activeRunId}</span>
            </div>
          )}
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Backend Host</span>
            <span className="text-green-500 font-medium flex items-center">
              <span className="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></span>
              PORT 8000 CONNECTED
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Run Status</span>
            <span className={`font-semibold uppercase px-2 py-0.5 rounded text-center text-xs ${
              runStatus === "COMPLETED" ? "bg-green-500/20 text-green-400 border border-green-500/30" :
              runStatus === "RUNNING" ? "bg-primary/20 text-primary border border-primary/30 animate-pulse" :
              runStatus === "FAILED" ? "bg-red-500/20 text-red-400 border border-red-500/30" :
              runStatus === "CANCELLED" ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" :
              "bg-[#1f1f1f] text-muted"
            }`}>
              {runStatus}
            </span>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-3 gap-6 overflow-hidden">

        {/* LEFT / CENTER: MEDIA WORKSPACE & TRANSFORM INSPECTOR (2 COLUMNS) */}
        <div className="xl:col-span-2 flex flex-col space-y-6 overflow-y-auto pr-1">

          {/* MEDIA WORKSPACE */}
          <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold tracking-wide flex items-center text-muted uppercase font-mono">
                <ImageIcon className="w-4 h-4 mr-2 text-primary" />
                Media-to-Media Workspace
              </h2>
              <div className="flex items-center space-x-2">
                <button
                  onClick={loadBaselineFixtures}
                  className="text-[11px] bg-[#1a1a1a] hover:bg-[#262626] border border-[#333] px-2.5 py-1 rounded font-mono text-muted hover:text-[#ededed] transition-colors"
                >
                  Reset Baseline Fixture (512×512)
                </button>
              </div>
            </div>

            {/* 4 CARDS: SECRET, COVER, STEGO, RECOVERED */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

              {/* CARD 1: SECRET MEDIA */}
              <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col relative group">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-semibold text-primary">1. SECRET MEDIA (PAYLOAD)</span>
                  <div className="flex items-center space-x-1.5">
                    {secretDims && (
                      <span className="text-[10px] font-mono text-primary bg-primary/10 border border-primary/20 px-1.5 py-0.5 rounded">
                        {secretDims.width}×{secretDims.height}
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-muted bg-[#1c1c1c] px-1.5 py-0.5 rounded">
                      {secretBytesSize > 0 ? `${secretBytesSize} B` : "No media"}
                    </span>
                  </div>
                </div>
                <div className="h-44 border border-dashed border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {secretPreview ? (
                    <img src={secretPreview} alt="Secret" className="max-h-full max-w-full object-contain p-2" />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-primary transition-colors" />
                      <span className="text-xs text-muted block">Click to upload secret payload</span>
                    </div>
                  )}
                  <input type="file" onChange={handleSecretUpload} disabled={running} accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" />
                </div>
                <div className="mt-2 text-[11px] font-mono text-muted truncate">
                  File: <span className="text-[#ededed]">{secretFilename}</span>
                </div>
              </div>

              {/* CARD 2: COVER MEDIA */}
              <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col relative group">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-semibold text-[#888]">2. COVER MEDIA (CARRIER)</span>
                  <span className="text-[10px] font-mono text-muted bg-[#1c1c1c] px-1.5 py-0.5 rounded">
                    {coverDims ? `${coverDims.width}×${coverDims.height} ${coverDims.mode || "RGB"}` : "Dynamic RGB"}
                  </span>
                </div>
                <div className="h-44 border border-dashed border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {coverPreview ? (
                    <img src={coverPreview} alt="Cover" className="max-h-full max-w-full object-contain p-2" />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-[#888] transition-colors" />
                      <span className="text-xs text-muted block">Click to upload carrier (PNG, even dims)</span>
                    </div>
                  )}
                  <input type="file" onChange={handleCoverUpload} disabled={running} accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" />
                </div>
                <div className="mt-2 text-[11px] font-mono text-muted truncate">
                  Carrier: <span className="text-[#ededed]">{coverFilename}</span>
                </div>
              </div>

              {/* CARD 3: STEGO MEDIA */}
              <div
                onClick={() => {
                  if (stegoImage) setIsStegoInspectorOpen(true);
                }}
                className={`border rounded-lg p-3 bg-[#0d0d0d] flex flex-col transition-all group ${
                  stegoImage
                    ? "border-green-500/40 hover:border-green-400 cursor-pointer shadow-sm hover:shadow-green-500/10"
                    : "border-[#262626]"
                }`}
                title={stegoImage ? "Click to inspect actual embedding locations and pixel differences" : "Awaiting embedding"}
              >
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="text-xs font-mono font-semibold text-green-400">3. STEGO MEDIA (RECONSTRUCTED)</span>
                    {stegoImage && (
                      <span className="text-[9px] bg-green-500/20 text-green-300 border border-green-500/30 px-1 py-0.2 rounded flex items-center font-mono">
                        <Eye className="w-2.5 h-2.5 mr-0.5" /> INSPECT
                      </span>
                    )}
                  </div>
                  {psnr !== null && (
                    <span className="text-[10px] font-mono text-green-400 bg-green-500/10 border border-green-500/20 px-1.5 py-0.5 rounded">
                      PSNR: {psnr} dB
                    </span>
                  )}
                </div>
                <div className="h-44 border border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {stegoImage ? (
                    <>
                      <img src={stegoImage} alt="Stego" className="max-h-full max-w-full object-contain p-2" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity backdrop-blur-[1px]">
                        <span className="text-xs font-mono bg-black/85 border border-primary/50 text-primary px-3 py-1.5 rounded flex items-center shadow-lg font-bold">
                          <Eye className="w-3.5 h-3.5 mr-1.5" /> Open Stego Inspector (5 Views)
                        </span>
                      </div>
                    </>
                  ) : runStatus === "RUNNING" ? (
                    <div className="text-center p-4 text-muted">
                      <Loader2 className="w-6 h-6 animate-spin mx-auto mb-1 text-primary" />
                      <span className="text-xs block font-mono">Generating stego container...</span>
                    </div>
                  ) : runStatus === "COMPLETED" || psnr !== null ? (
                    <div className="text-center p-4 text-green-400 font-mono">
                      <CheckCircle2 className="w-6 h-6 mx-auto mb-1 text-green-400" />
                      <span className="text-xs block">Stego Container Ready ({coverDims?.width || 512}×{coverDims?.height || 512})</span>
                    </div>
                  ) : runStatus === "FAILED" ? (
                    <div className="text-center p-4 text-red-400 font-mono">
                      <AlertTriangle className="w-6 h-6 mx-auto mb-1 text-red-500 opacity-60" />
                      <span className="text-xs block">Embedding Failed</span>
                    </div>
                  ) : runStatus === "CANCELLED" ? (
                    <div className="text-center p-4 text-muted font-mono">
                      <Lock className="w-6 h-6 mx-auto mb-1 opacity-40" />
                      <span className="text-xs block">Run Cancelled</span>
                    </div>
                  ) : (
                    <div className="text-center p-4 text-muted">
                      <Lock className="w-6 h-6 mx-auto mb-1 opacity-40" />
                      <span className="text-xs block">Awaiting embedding & IDWT</span>
                    </div>
                  )}
                </div>
                <div className="mt-2 flex justify-between text-[11px] font-mono text-muted">
                  <span>SSIM: <span className="text-[#ededed]">{ssim !== null ? ssim : "N/A"}</span></span>
                  <span>Container: <span className="text-[#ededed]">8-bit PNG</span></span>
                </div>
              </div>

              {/* CARD 4: RECOVERED SECRET */}
              <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-semibold text-cyan-400">4. RECOVERED SECRET MEDIA</span>
                  {exactMatch !== null && (
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded flex items-center ${
                      exactMatch
                        ? "bg-green-500/20 text-green-400 border border-green-500/30"
                        : "bg-red-500/20 text-red-400 border border-red-500/30"
                    }`}>
                      {exactMatch ? <CheckCircle2 className="w-3 h-3 mr-1" /> : <XCircle className="w-3 h-3 mr-1" />}
                      {exactMatch ? "EXACT MATCH" : "MISMATCH"}
                    </span>
                  )}
                </div>
                <div className="h-44 border border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {recoveredImage ? (
                    <img src={recoveredImage} alt="Recovered Secret" className="max-h-full max-w-full object-contain p-2" />
                  ) : runStatus === "RUNNING" ? (
                    <div className="text-center p-4 text-muted">
                      <Loader2 className="w-6 h-6 animate-spin mx-auto mb-1 text-cyan-400" />
                      <span className="text-xs block font-mono">Extracting & verifying secret...</span>
                    </div>
                  ) : runStatus === "FAILED" ? (
                    <div className="text-center p-4 text-red-400 font-mono">
                      <XCircle className="w-6 h-6 mx-auto mb-1 text-red-500 opacity-60" />
                      <span className="text-xs block">Recovery Incomplete / Failed</span>
                    </div>
                  ) : runStatus === "CANCELLED" ? (
                    <div className="text-center p-4 text-muted font-mono">
                      <Zap className="w-6 h-6 mx-auto mb-1 opacity-40" />
                      <span className="text-xs block">Run Cancelled</span>
                    </div>
                  ) : (
                    <div className="text-center p-4 text-muted">
                      <Zap className="w-6 h-6 mx-auto mb-1 opacity-40" />
                      <span className="text-xs block">Awaiting receiver recovery</span>
                    </div>
                  )}
                </div>
                <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-muted">
                  <span>
                    Status: <span className={exactMatch ? "text-green-400 font-semibold" : "text-[#ededed]"}>
                      {exactMatch === true ? "100% BYTE EQUALITY" : exactMatch === false ? "VERIFICATION FAILED" : "STANDBY"}
                    </span>
                  </span>
                  {recoveredImage && (
                    <button
                      onClick={() => setIsSecretRecoveryOpen(true)}
                      className="text-[10px] text-primary hover:text-primary/80 font-semibold underline flex items-center"
                    >
                      Compare Side-by-Side
                    </button>
                  )}
                </div>
              </div>

            </div>

            {/* CAPACITY WARNING IF PAYLOAD EXCEEDS CARRIER LIMIT */}
            {secretBytesSize > capacityLimits.max_protected_bytes && (
              <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start space-x-2.5 font-mono text-xs text-red-400">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 text-red-400 mt-0.5" />
                <div>
                  <strong className="block text-red-300">Payload Exceeds Carrier Capacity</strong>
                  <span>
                    Selected secret payload ({secretBytesSize.toLocaleString()} B) exceeds maximum protected capacity ({capacityLimits.max_protected_bytes.toLocaleString()} B) for this {coverDims?.width || 512}×{coverDims?.height || 512} carrier. Pipeline embedding will be rejected by the hard capacity gate. Select a larger carrier or smaller secret payload.
                  </span>
                </div>
              </div>
            )}

            {/* ACTION CONTROLS */}
            <div className="mt-5 pt-4 border-t border-[#262626] flex items-center justify-between">
              <span className="text-xs font-mono text-muted">
                Pipeline execution runs 100% on the genuine Python backend.
              </span>
              <div className="flex space-x-3">
                <button
                  onClick={handleCancel}
                  disabled={!running}
                  className="px-4 py-2 text-xs font-mono font-semibold border border-[#333] rounded hover:bg-[#1f1f1f] disabled:opacity-30 flex items-center"
                >
                  <Square className="w-3.5 h-3.5 mr-1.5" /> CANCEL
                </button>
                <button
                  onClick={handleRun}
                  disabled={running}
                  className="px-5 py-2 text-xs font-mono font-semibold bg-primary hover:bg-primary/90 text-white rounded disabled:opacity-40 flex items-center shadow-sm"
                >
                  <Play className="w-3.5 h-3.5 mr-1.5 fill-current" /> RUN PHASE 2 PIPELINE
                </button>
              </div>
            </div>
          </div>

          {/* REAL DWT SUBBAND VISUALIZER & TRANSFORM INSPECTOR */}
          <DwtSubbandInspector
            dwtSubbands={dwtSubbands}
            modifiedHhImage={modifiedHhImage}
            coverImageUrl={coverPreview}
            stegoImageUrl={stegoImage}
            diagnostics={diagnostics}
            activeRunId={activeRunId}
          />

          {/* CAPACITY HARD GATE INSPECTOR */}
          <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-sm font-semibold tracking-wide flex items-center text-muted uppercase font-mono">
                <Activity className="w-4 h-4 mr-2 text-primary" />
                Carrier Capacity Accounting & Hard Gate
              </h2>
              {capacityInfo && (
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                  capacityInfo.fits
                    ? "bg-green-500/20 text-green-400 border border-green-500/30"
                    : "bg-red-500/20 text-red-400 border border-red-500/30"
                }`}>
                  STATUS: {capacityInfo.status}
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs mb-3">
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Raw Bit Budget</span>
                <span className="text-sm font-bold text-[#ededed]">
                  {capacityLimits.raw_secret_capacity_bytes.toLocaleString()} B
                </span>
                <span className="text-[9px] text-muted block mt-0.5">
                  {capacityLimits.subband_width}×{capacityLimits.subband_height} bits = 1 HH channel
                </span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Max Protected</span>
                <span className="text-sm font-bold text-[#ededed]">
                  {capacityLimits.max_protected_bytes.toLocaleString()} B
                </span>
                <span className="text-[9px] text-muted block mt-0.5">
                  {capacityLimits.raw_secret_capacity_bytes.toLocaleString()} B - 16B Carrier Header
                </span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Usable Plaintext</span>
                <span className="text-sm font-bold text-primary">
                  ~{Math.max(0, capacityLimits.max_protected_bytes - 48 - 28 - 100).toLocaleString()} B
                </span>
                <span className="text-[9px] text-muted block mt-0.5">After RS parity + AES + Wire</span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Protected Payload</span>
                <span className={`text-sm font-bold ${
                  capacityInfo && !capacityInfo.fits ? "text-red-400" : "text-[#ededed]"
                }`}>
                  {capacityInfo ? `${capacityInfo.protected_payload_bytes} B` : (secretBytesSize > 0 ? `${secretBytesSize} B (raw)` : "-- B")}
                </span>
                <span className="text-[9px] text-muted block mt-0.5">
                  {capacityInfo ? `Remaining: ${capacityInfo.remaining_capacity_bytes} B` : (secretBytesSize > 0 ? `Max: ${capacityLimits.max_protected_bytes.toLocaleString()} B` : "Awaiting check")}
                </span>
              </div>
            </div>

            {/* Capacity Visual Progress Bar */}
            <div className="w-full bg-[#1c1c1c] h-2 rounded-full overflow-hidden border border-[#2b2b2b]">
              <div
                className={`h-full transition-all duration-500 ${
                  capacityInfo && !capacityInfo.fits
                    ? "bg-red-500"
                    : secretBytesSize > capacityLimits.max_protected_bytes
                    ? "bg-red-500"
                    : "bg-primary"
                }`}
                style={{
                  width: capacityInfo
                    ? `${Math.min(100, (capacityInfo.protected_payload_bytes / capacityInfo.max_protected_bytes) * 100)}%`
                    : secretBytesSize > 0
                    ? `${Math.min(100, (secretBytesSize / capacityLimits.max_protected_bytes) * 100)}%`
                    : "0%"
                }}
              />
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: REAL PIPELINE TIMELINE & EXECUTION LOG */}
        <div className="flex flex-col space-y-6 overflow-hidden">

          {/* PIPELINE TIMELINE */}
          <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm flex-1 overflow-y-auto">
            <h2 className="text-sm font-semibold tracking-wide text-muted uppercase font-mono mb-4 flex items-center">
              <Zap className="w-4 h-4 mr-2 text-primary" />
              Transform-Domain Pipeline Stages
            </h2>
            <div className="space-y-2">
              {STAGES.map((s, idx) => {
                const state = stages[s];
                let borderCol = "border-[#212121]";
                let textCol = "text-muted";
                let dotCol = "bg-[#333]";

                if (state.status === "processing") {
                  borderCol = "border-primary/50 bg-primary/5";
                  textCol = "text-primary";
                  dotCol = "bg-primary animate-pulse";
                } else if (state.status === "complete") {
                  borderCol = "border-green-500/30 bg-green-500/5";
                  textCol = "text-green-400";
                  dotCol = "bg-green-500";
                } else if (state.status === "failed") {
                  borderCol = "border-red-500/40 bg-red-500/10";
                  textCol = "text-red-400";
                  dotCol = "bg-red-500";
                }

                return (
                  <div key={s} className={`p-2.5 rounded border ${borderCol} transition-colors flex items-start text-xs font-mono`}>
                    <div className="mt-1 mr-2.5 flex-shrink-0">
                      <div className={`w-2.5 h-2.5 rounded-full ${dotCol}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-center">
                        <span className={`font-semibold ${textCol}`}>{idx + 1}. {STAGE_LABELS[s]}</span>
                        <span className={`text-[10px] uppercase font-bold ${textCol}`}>{state.status}</span>
                      </div>
                      {state.message && (
                        <p className="text-[11px] text-muted mt-0.5 truncate">{state.message}</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* REAL TERMINAL LOG */}
          <div className="bg-[#050505] border border-[#262626] rounded-xl p-4 shadow-sm h-64 flex flex-col font-mono text-xs">
            <div className="flex justify-between items-center mb-2 pb-1 border-b border-[#1c1c1c]">
              <span className="text-[11px] font-semibold text-muted uppercase">Execution Telemetry Log</span>
              <button onClick={() => setLogs([])} className="text-[10px] text-muted hover:text-[#ededed]">CLEAR</button>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1 text-green-500/90 font-mono select-text" ref={logRef}>
              {logs.length === 0 ? (
                <span className="text-muted/60">Awaiting laboratory trigger...</span>
              ) : (
                logs.map((line, i) => <div key={i}>{line}</div>)
              )}
            </div>
          </div>

        </div>

      </main>

      {/* STEGO-MEDIA EMBEDDING INSPECTOR MODAL */}
      <StegoInspectorModal
        isOpen={isStegoInspectorOpen}
        onClose={() => setIsStegoInspectorOpen(false)}
        stegoImageUrl={stegoImage}
        coverImageUrl={coverPreview}
        diagnostics={diagnostics}
        pixelDiffMetrics={pixelDiffMetrics}
        psnr={psnr}
        ssim={ssim}
      />

      {/* SECRET RECOVERY VERIFICATION MODAL */}
      <SecretRecoveryModal
        isOpen={isSecretRecoveryOpen}
        onClose={() => setIsSecretRecoveryOpen(false)}
        originalImageUrl={secretPreview}
        recoveredImageUrl={recoveredImage}
        originalFilename={secretFilename}
        originalBytes={secretBytesSize}
        recoveredBytes={recoveredBytes}
        originalDims={originalDims}
        recoveredDims={recoveredDims}
        exactMatch={exactMatch}
      />

    </div>
  );
}
