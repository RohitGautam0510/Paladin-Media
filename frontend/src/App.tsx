import React, { useState, useEffect, useRef } from 'react';
import type {
  PipelineEvent,
  PipelineStage,
  StageStatus,
  DefaultFixtures,
  BackendCapabilities,
  SustainabilityMetrics,
} from './services/backendAdapter';
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
  Loader2,
  Cpu,
  Layers,
  Unlock,
  Key,
  FileText,
  Download,
  FileImage
} from 'lucide-react';
import { StegoInspectorModal, type DiagnosticsMetadata, type PixelDiffMetrics } from './components/StegoInspectorModal';
import { DwtSubbandInspector } from './components/DwtSubbandInspector';
import { SecretRecoveryModal } from './components/SecretRecoveryModal';
import { SustainabilityWidget } from './components/SustainabilityWidget';

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
    reference_metadata_capacity_bytes?: number;
    header_wire_bytes?: number;
    ecc_profile?: string;
  }>({
    raw_secret_capacity_bytes: 6578,
    max_protected_bytes: 8168,
    subband_width: 256,
    subband_height: 256,
    reference_metadata_capacity_bytes: 6576,
    header_wire_bytes: 24,
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

  // Runtime Capabilities, Model Selection & Telemetry
  const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("neural_spatial_onnx");
  const [selectedPrecision, setSelectedPrecision] = useState<string>("fp32");
  const [selectedProtocolVersion, setSelectedProtocolVersion] = useState<number>(2);
  const [telemetry, setTelemetry] = useState<SustainabilityMetrics | null>(null);
  const [activeTab, setActiveTab] = useState<"EXPERIMENT" | "STANDALONE_EXTRACT">("EXPERIMENT");

  // Standalone Extraction state
  const [extractFile, setExtractFile] = useState<File | null>(null);
  const [extractPreview, setExtractPreview] = useState<string | null>(null);
  const [extractFilename, setExtractFilename] = useState<string>("No file chosen");
  const [extractFileSize, setExtractFileSize] = useState<number>(0);
  const [extractDims, setExtractDims] = useState<{ width: number; height: number } | null>(null);
  const [extractKeyHex, setExtractKeyHex] = useState<string>("");
  const [extracting, setExtracting] = useState<boolean>(false);
  const [extractResult, setExtractResult] = useState<any>(null);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [extractedMediaUrl, setExtractedMediaUrl] = useState<string | null>(null);
  const [defaultFixtures, setDefaultFixtures] = useState<DefaultFixtures | null>(null);

  // Authentication & BFF Session state
  const [authSession, setAuthSession] = useState<{
    authenticated: boolean;
    requires_auth?: boolean;
    user?: string | null;
    roles?: string[];
    oidc_configured?: boolean;
    oidc_issuer?: string | null;
    idp_gate_status?: string;
  }>({ authenticated: false });
  const [loginModalOpen, setLoginModalOpen] = useState(false);
  const [loginError] = useState<string | null>(null);

  const logRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, `[${timestamp}] ${msg}`]);
  };

  const loadCapabilities = () => {
    backend.fetchCapabilities()
      .then((caps: BackendCapabilities) => {
        setCapabilities(caps);
        if (caps.active_engine) {
          setSelectedModel(caps.active_engine);
        }
        if (caps.default_protocol_version) {
          setSelectedProtocolVersion(caps.default_protocol_version);
        }
        if (caps.default_precision) {
          setSelectedPrecision(caps.default_precision);
        }
        addLog(`Loaded backend capabilities: Model=${caps.active_engine}, Telemetry=${caps.telemetry.source} (${caps.telemetry.is_estimated ? "Estimated" : "Direct"})`);
      })
      .catch((err) => {
        console.warn("Could not load backend capabilities:", err);
      });
  };

  const loadBaselineFixtures = () => {
    backend.fetchDefaultFixtures()
      .then((fixtures: DefaultFixtures) => {
        setDefaultFixtures(fixtures);
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
          backend.fetchCarrierCapacity(cW, cH, selectedProtocolVersion)
            .then(cap => setCapacityLimits(cap))
            .catch(() => {});
        }
        addLog("Preloaded baseline fixtures: secret_small.png (24x24) & 0001x2.png (512x512)");
      })
      .catch((err) => {
        console.warn("Could not load default fixtures automatically:", err);
      });
  };

  const checkAuthAndInit = async () => {
    try {
      const session = await backend.checkSession();
      setAuthSession(session);
      if (session.authenticated || !session.requires_auth) {
        setLoginModalOpen(false);
        loadCapabilities();
        loadBaselineFixtures();
      } else {
        setLoginModalOpen(true);
      }
    } catch (err) {
      console.warn("Session check fallback:", err);
      loadCapabilities();
      loadBaselineFixtures();
    }
  };

  const handleOidcLogin = () => {
    window.location.href = backend.getOidcLoginUrl();
  };

  const handleLogout = async () => {
    try {
      await backend.logout();
      setAuthSession({ authenticated: false, requires_auth: true });
      addLog("Operator session terminated. Logged out.");
      setLoginModalOpen(true);
    } catch (err) {
      console.warn("Logout error:", err);
    }
  };

  useEffect(() => {
    checkAuthAndInit();
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const processSecretFile = async (file: File) => {
    const maxUploadBytes = capabilities?.max_upload_size_bytes || (10 * 1024 * 1024);
    if (file.size > maxUploadBytes) {
      addLog(`ERROR: Secret file size (${(file.size / 1024 / 1024).toFixed(2)} MB) exceeds maximum upload limit of ${(maxUploadBytes / 1024 / 1024).toFixed(2)} MB.`);
      return;
    }
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
  };

  const handleSecretUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processSecretFile(e.target.files[0]);
    }
  };

  const processCoverFile = async (file: File) => {
    const maxUploadBytes = capabilities?.max_upload_size_bytes || (10 * 1024 * 1024);
    if (file.size > maxUploadBytes) {
      addLog(`ERROR: Carrier file size (${(file.size / 1024 / 1024).toFixed(2)} MB) exceeds maximum upload limit of ${(maxUploadBytes / 1024 / 1024).toFixed(2)} MB.`);
      return;
    }
    if (file.type && file.type !== "image/png" && !file.name.toLowerCase().endsWith(".png")) {
      addLog(`ERROR: Unsupported carrier format "${file.type || file.name}". PaladinMedia strictly requires lossless 8-bit RGB PNG carriers. Lossy formats (JPEG/WebP) are not supported.`);
      return;
    }
    setCoverFilename(file.name);
    const url = URL.createObjectURL(file);
    setCoverPreview(url);
    addLog(`Selected cover carrier image: ${file.name} (${file.size} bytes)`);

    const cImg = new Image();
    cImg.onload = () => {
      const w = cImg.naturalWidth;
      const h = cImg.naturalHeight;
      const maxDim = capabilities?.max_carrier_dimension || 2048;

      if (w > maxDim || h > maxDim) {
        addLog(`ERROR: Carrier dimensions ${w}×${h} exceed maximum permitted ${maxDim}×${maxDim} for this deployment.`);
        return;
      }

      setCoverDims({ width: w, height: h, format: "PNG", mode: "RGB" });
      setIsDatasetCustom(true);
      addLog(`Cover image dimensions: ${w}×${h} RGB`);

      if (w < 32 || h < 32) {
        addLog(`WARNING: Carrier dimensions ${w}×${h} are below minimum 32×32.`);
      } else if (w % 2 !== 0 || h % 2 !== 0) {
        addLog(`WARNING: Carrier dimensions ${w}×${h} are odd. DWT decomposition requires even dimensions.`);
      } else {
        backend.fetchCarrierCapacity(w, h, selectedProtocolVersion)
          .then(cap => {
            setCapacityLimits(cap);
            addLog(`Carrier capacity: ${cap.raw_secret_capacity_bytes.toLocaleString()} B raw secret, ${cap.max_protected_bytes.toLocaleString()} B max protected.`);
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
  };

  const handleCoverUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processCoverFile(e.target.files[0]);
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
    setTelemetry(null);
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
    addLog(`--- RUN INITIALIZED: Model=${selectedModel}, Protocol=v${selectedProtocolVersion} ---`);

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
        if (event.metadata?.telemetry) {
          setTelemetry(event.metadata.telemetry);
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
          if (event.metadata.telemetry) {
            setTelemetry(event.metadata.telemetry);
          }
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
      },
      undefined,
      selectedModel,
      selectedProtocolVersion,
      selectedPrecision
    );
  };

  const handleTransferStegoToExtract = async () => {
    if (!stegoImage) {
      addLog("WARNING: No generated stego container available from experiment yet.");
      return;
    }
    try {
      addLog("Transferring generated stego carrier to standalone extraction pipeline...");
      const res = await fetch(stegoImage);
      const blob = await res.blob();
      const filename = `stego_carrier_${activeRunId || "experiment"}.png`;
      const file = new File([blob], filename, { type: "image/png" });
      setExtractFile(file);
      setExtractFilename(filename);
      setExtractFileSize(file.size);
      setExtractPreview(stegoImage);
      setExtractResult(null);
      setExtractError(null);
      setExtractedMediaUrl(null);
      if (coverDims) {
        setExtractDims({ width: coverDims.width, height: coverDims.height });
      }
      setActiveTab("STANDALONE_EXTRACT");
      addLog(`Stego container transferred: ${coverDims?.width || 512}×${coverDims?.height || 512} (${file.size} bytes). Ready for extraction.`);
    } catch (err: any) {
      addLog(`ERROR: Failed to transfer stego image: ${err.message}`);
    }
  };

  const handleLoadSampleStego = async () => {
    try {
      addLog("Loading pre-encoded sample stego container fixture...");
      let fixture = defaultFixtures?.stego;
      if (!fixture?.data_url) {
        const fixtures = await backend.fetchDefaultFixtures();
        setDefaultFixtures(fixtures);
        fixture = fixtures.stego;
      }
      if (!fixture?.data_url) {
        addLog("ERROR: Sample stego fixture is not available on this server.");
        return;
      }
      const res = await fetch(fixture.data_url);
      const blob = await res.blob();
      const file = new File([blob], fixture.name || "sample_stego_carrier_512.png", { type: "image/png" });
      setExtractFile(file);
      setExtractFilename(file.name);
      setExtractFileSize(file.size);
      setExtractPreview(fixture.data_url);
      setExtractResult(null);
      setExtractError(null);
      setExtractedMediaUrl(null);
      setExtractDims({ width: fixture.width || 512, height: fixture.height || 512 });
      addLog(`Loaded sample pre-encoded stego carrier: ${file.name} (${fixture.width || 512}×${fixture.height || 512}, ${file.size} bytes)`);
    } catch (err: any) {
      addLog(`ERROR: Failed to load sample stego carrier: ${err.message}`);
    }
  };

  const processExtractCarrierFile = (file: File) => {
    if (file.type && file.type !== "image/png" && !file.name.toLowerCase().endsWith(".png")) {
      addLog(`ERROR: Carrier must be a lossless PNG stego container. Got "${file.type || file.name}".`);
      setExtractError("Unsupported carrier format: PaladinMedia requires an 8-bit RGB lossless PNG carrier.");
      return;
    }
    setExtractFile(file);
    setExtractFilename(file.name);
    setExtractFileSize(file.size);
    const url = URL.createObjectURL(file);
    setExtractPreview(url);
    setExtractResult(null);
    setExtractError(null);
    setExtractedMediaUrl(null);
    const img = new Image();
    img.onload = () => {
      setExtractDims({ width: img.naturalWidth, height: img.naturalHeight });
      addLog(`Carrier for extraction loaded: ${img.naturalWidth}×${img.naturalHeight} (${file.size} bytes)`);
    };
    img.src = url;
  };

  const handleExtractCarrierUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processExtractCarrierFile(e.target.files[0]);
    }
  };

  const handleStandaloneExtract = async () => {
    if (!extractFile) {
      addLog("ERROR: Please select or upload a stego carrier PNG first.");
      return;
    }
    setExtracting(true);
    setExtractError(null);
    setExtractResult(null);
    setExtractedMediaUrl(null);
    addLog(`--- STANDALONE EXTRACTION TRIGGERED: File=${extractFilename}, Model=${selectedModel}, Precision=${selectedPrecision} ---`);

    try {
      const res = await backend.extractMedia(
        extractFile,
        extractKeyHex.trim() || undefined,
        selectedModel,
        selectedPrecision
      );
      setExtractResult(res);
      addLog(`Extraction succeeded! Protocol v${res.wire_protocol_version}, Parity=${res.ecc_parity}, Recovered=${res.recovered_bytes} B`);
      if (res.recovered_session_id) {
        const mediaUrl = await backend.fetchSessionMedia(res.recovered_session_id);
        setExtractedMediaUrl(mediaUrl);
        setRecoveredImage(mediaUrl);
        setRecoveredBytes(res.recovered_bytes);
      }
    } catch (err: any) {
      const errMsg = err.message || "Extraction failed";
      setExtractError(errMsg);
      addLog(`EXTRACTION FAILED: ${errMsg}`);
    } finally {
      setExtracting(false);
    }
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
      <header className="border-b border-[#262626] bg-[#121212] px-6 py-4 flex flex-col xl:flex-row items-start xl:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded bg-primary/20 border border-primary/40 flex items-center justify-center">
            <Shield className="text-primary w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-wider font-mono">PALADINMEDIA</h1>
              <span className="text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded font-mono font-semibold">
                POST-QUANTUM NEURAL STEGANOGRAPHY
              </span>
            </div>
            <p className="text-xs text-muted font-mono">ACS-Sub-INN v3 + Wavelet Core + Reed-Solomon GF(2^8)</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-6 text-xs font-mono">
          {/* Engine Selector */}
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase flex items-center">
              <Cpu className="w-3 h-3 mr-1 text-primary" /> Modulation Engine
            </span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={running}
              aria-label="Modulation Engine"
              className="bg-[#1a1a1a] border border-[#333] text-[#ededed] px-2 py-1 rounded text-xs font-mono focus:border-primary focus:outline-none"
            >
              <option value="neural_spatial_onnx">Neural Spatial ONNX (Qualified)</option>
              <option value="acs_sub_inn_onnx">ACS-Sub-INN Subband ONNX</option>
              <option value="acs_sub_inn_pytorch">ACS-Sub-INN PyTorch Ref</option>
              <option value="parity_classical">Deterministic Parity (Phase 2)</option>
            </select>
          </div>

          {/* Protocol Version */}
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase flex items-center">
              <Layers className="w-3 h-3 mr-1 text-cyan-400" /> Wire Protocol
            </span>
            <select
              value={selectedProtocolVersion}
              onChange={(e) => {
                const newVer = Number(e.target.value);
                setSelectedProtocolVersion(newVer);
                if (coverDims) {
                  backend.fetchCarrierCapacity(coverDims.width, coverDims.height, newVer)
                    .then(cap => {
                      setCapacityLimits(cap);
                      addLog(`Protocol changed to v${newVer}: Capacity updated to ${cap.raw_secret_capacity_bytes.toLocaleString()} B raw, ${cap.max_protected_bytes.toLocaleString()} B max protected.`);
                    })
                    .catch(err => addLog(`Capacity calculation error: ${err.message}`));
                }
              }}
              disabled={running || (capabilities?.supported_protocol_versions?.length ?? 1) <= 1}
              aria-label="Wire Protocol"
              className="bg-[#1a1a1a] border border-[#333] text-[#ededed] px-2 py-1 rounded text-xs font-mono focus:border-cyan-400 focus:outline-none"
            >
              {(capabilities?.supported_protocol_versions || [2]).map((v) => (
                <option key={v} value={v}>
                  Protocol v{v} {v === 3 ? "(v3-MR RS(255,191))" : v === 2 ? "(RS(24,16) Production Default)" : "(Legacy)"}
                </option>
              ))}
            </select>
          </div>

          {/* Precision Selector */}
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase flex items-center">
              <Cpu className="w-3 h-3 mr-1 text-emerald-400" /> Precision
            </span>
            <select
              value={selectedPrecision}
              onChange={(e) => setSelectedPrecision(e.target.value)}
              disabled={running || extracting}
              aria-label="Model Precision"
              className="bg-[#1a1a1a] border border-[#333] text-[#ededed] px-2 py-1 rounded text-xs font-mono focus:border-emerald-400 focus:outline-none"
            >
              <option value="fp32">FP32 (Reference Precision)</option>
              <option value="int8">INT8 QDQ (Static Quantized)</option>
            </select>
          </div>

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
            <span className="text-muted text-[10px] uppercase">Backend Status</span>
            <span className="text-green-500 font-medium flex items-center">
              <span className="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></span>
              {capabilities?.status === "OPERATIONAL" ? "OPERATIONAL (PORT 8000)" : "CONNECTED"}
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

          {/* Operator Auth / BFF Session Status */}
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Session Auth</span>
            {authSession.requires_auth ? (
              authSession.authenticated ? (
                <div className="flex items-center space-x-2">
                  <span className="text-cyan-400 font-medium flex items-center text-xs">
                    <Lock className="w-3 h-3 mr-1 text-cyan-400" />
                    Operator
                  </span>
                  <button
                    onClick={handleLogout}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-red-950/40 border border-red-800/40 text-red-400 hover:bg-red-900/60 transition-colors"
                    title="Terminate active session"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setLoginModalOpen(true)}
                  className="text-xs text-amber-400 font-medium flex items-center hover:underline"
                >
                  <Key className="w-3 h-3 mr-1 text-amber-400" />
                  Sign In
                </button>
              )
            ) : (
              <span className="text-muted font-medium text-xs flex items-center">
                <Shield className="w-3 h-3 mr-1 text-muted" />
                Local Dev
              </span>
            )}
          </div>
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <div className="bg-[#0e0e0e] border-b border-[#262626] px-6 py-2.5 flex items-center space-x-3 text-xs font-mono">
        <button
          onClick={() => setActiveTab("EXPERIMENT")}
          className={`px-3.5 py-1.5 rounded transition-all font-semibold flex items-center ${
            activeTab === "EXPERIMENT"
              ? "bg-primary/20 text-primary border border-primary/40 shadow-sm"
              : "text-muted hover:text-[#ededed] hover:bg-[#1a1a1a]"
          }`}
        >
          <Play className="w-3.5 h-3.5 mr-1.5 fill-current" />
          Live Steganography Experiment
        </button>
        <button
          onClick={() => setActiveTab("STANDALONE_EXTRACT")}
          className={`px-3.5 py-1.5 rounded transition-all font-semibold flex items-center ${
            activeTab === "STANDALONE_EXTRACT"
              ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm"
              : "text-muted hover:text-[#ededed] hover:bg-[#1a1a1a]"
          }`}
        >
          <Unlock className="w-3.5 h-3.5 mr-1.5" />
          Standalone Carrier Extraction
        </button>
      </div>

      {/* MAIN CONTAINER */}
      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-3 gap-6 overflow-hidden">
        
        {/* LEFT / CENTER: MEDIA WORKSPACE & TRANSFORM INSPECTOR (2 COLUMNS) */}
        <div className="xl:col-span-2 flex flex-col space-y-6 overflow-y-auto pr-1">
          {activeTab === "EXPERIMENT" ? (
            <>
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
                <div
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!running && e.dataTransfer.files && e.dataTransfer.files[0]) {
                      processSecretFile(e.dataTransfer.files[0]);
                    }
                  }}
                  className="h-44 border border-dashed border-[#2b2b2b] hover:border-primary/60 rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707] transition-colors"
                >
                  {secretPreview ? (
                    <img src={secretPreview} alt="Secret" className="max-h-full max-w-full object-contain p-2" />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-primary transition-colors" />
                      <span className="text-xs text-muted block">Click or drag & drop to upload secret payload</span>
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
                <div
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!running && e.dataTransfer.files && e.dataTransfer.files[0]) {
                      processCoverFile(e.dataTransfer.files[0]);
                    }
                  }}
                  className="h-44 border border-dashed border-[#2b2b2b] hover:border-[#666] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707] transition-colors"
                >
                  {coverPreview ? (
                    <img src={coverPreview} alt="Cover" className="max-h-full max-w-full object-contain p-2" />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-[#888] transition-colors" />
                      <span className="text-xs text-muted block">Click or drag & drop to upload carrier (PNG, even dims)</span>
                    </div>
                  )}
                  <input type="file" onChange={handleCoverUpload} disabled={running} accept="image/png" className="absolute inset-0 opacity-0 cursor-pointer" />
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
                  <div className="flex items-center space-x-1.5">
                    {psnr !== null && (
                      <span className="text-[10px] font-mono text-green-400 bg-green-500/10 border border-green-500/20 px-1.5 py-0.5 rounded">
                        PSNR: {psnr} dB
                      </span>
                    )}
                    {stegoImage && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const a = document.createElement("a");
                          a.href = stegoImage;
                          a.download = `stego_carrier_${activeRunId || "export"}.png`;
                          a.click();
                        }}
                        className="text-[10px] font-mono font-semibold bg-primary/20 text-primary hover:bg-primary/30 border border-primary/40 px-1.5 py-0.5 rounded flex items-center transition-colors"
                        title="Download untouched 8-bit RGB Stego PNG"
                      >
                        <Download className="w-2.5 h-2.5 mr-1" /> PNG
                      </button>
                    )}
                  </div>
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
                <div className="mt-2 flex justify-between items-center text-[11px] font-mono text-muted">
                  <span>SSIM: <span className="text-[#ededed]">{ssim !== null ? ssim : "N/A"}</span></span>
                  {stegoImage ? (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleTransferStegoToExtract();
                      }}
                      className="text-[10px] text-cyan-400 hover:text-cyan-300 font-semibold underline flex items-center"
                      title="Load this stego container directly into the Standalone Extraction tab"
                    >
                      Extract Hidden Secret &rarr;
                    </button>
                  ) : (
                    <span>Container: <span className="text-[#ededed]">8-bit PNG</span></span>
                  )}
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
                  <Play className="w-3.5 h-3.5 mr-1.5 fill-current" /> RUN STEGANOGRAPHY PIPELINE
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

          {/* SUSTAINABILITY & GREEN COMPUTING TELEMETRY */}
          <SustainabilityWidget telemetry={telemetry} isRunning={running} />

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
                  {capacityLimits.raw_secret_capacity_bytes.toLocaleString()} B raw - {capacityLimits.header_wire_bytes || 24}B Header
                </span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Usable Plaintext</span>
                <span className="text-sm font-bold text-primary">
                  {capacityLimits.reference_metadata_capacity_bytes ? `${capacityLimits.reference_metadata_capacity_bytes.toLocaleString()} B` : `${capacityLimits.raw_secret_capacity_bytes.toLocaleString()} B`}
                </span>
                <span className="text-[9px] text-muted block mt-0.5">{capacityLimits.reference_metadata_capacity_bytes ? "With 2B reference metadata" : "Zero metadata payload"}</span>
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
            </>
          ) : (
            <>
              {/* STANDALONE EXTRACTION CONTROLLER */}
              <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold tracking-wide flex items-center text-cyan-400 uppercase font-mono">
                    <Unlock className="w-4 h-4 mr-2" />
                    Standalone Carrier Extraction & Verification
                  </h2>
                  <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2 py-0.5 rounded font-mono font-semibold">
                    FAIL-CLOSED VERIFICATION
                  </span>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 mb-4 bg-[#0a0a0a] p-3 rounded-lg border border-[#262626]">
                  <div className="text-xs font-mono text-muted flex items-center space-x-1.5">
                    <span className="text-[#ededed] font-semibold">One-Click Test Actions:</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {stegoImage && (
                      <button
                        onClick={handleTransferStegoToExtract}
                        disabled={extracting}
                        className="px-2.5 py-1 text-xs font-mono bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 border border-cyan-500/40 rounded flex items-center transition-colors shadow-sm"
                        title="Load generated stego carrier from latest experiment run"
                      >
                        <Zap className="w-3.5 h-3.5 mr-1.5 text-cyan-400" /> Use Stego from Latest Run
                      </button>
                    )}
                    <button
                      onClick={handleLoadSampleStego}
                      disabled={extracting}
                      className="px-2.5 py-1 text-xs font-mono bg-[#1c1c1c] text-[#ededed] hover:bg-[#252525] border border-[#3a3a3a] rounded flex items-center transition-colors shadow-sm"
                      title="Load verified pre-encoded sample stego carrier fixture (512x512)"
                    >
                      <FileImage className="w-3.5 h-3.5 mr-1.5 text-primary" /> Load Sample Stego (512×512)
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* DROPZONE: STEGO CARRIER */}
                  <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col relative group">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-mono font-semibold text-cyan-400">STEGO CARRIER (PNG)</span>
                      <div className="flex items-center space-x-1.5">
                        {extractDims && (
                          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-1.5 py-0.5 rounded">
                            {extractDims.width}×{extractDims.height}
                          </span>
                        )}
                        <span className="text-[10px] font-mono text-muted bg-[#1c1c1c] px-1.5 py-0.5 rounded">
                          {extractFileSize > 0 ? `${extractFileSize} B` : "No file"}
                        </span>
                      </div>
                    </div>
                    <div
                      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                      onDrop={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!extracting && e.dataTransfer.files && e.dataTransfer.files[0]) {
                          processExtractCarrierFile(e.dataTransfer.files[0]);
                        }
                      }}
                      className="h-48 border border-dashed border-[#2b2b2b] hover:border-cyan-400/60 rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707] transition-colors"
                    >
                      {extractPreview ? (
                        <img src={extractPreview} alt="Stego Carrier" className="max-h-full max-w-full object-contain p-2" />
                      ) : (
                        <div className="text-center p-4">
                          <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-cyan-400 transition-colors" />
                          <span className="text-xs text-muted block">Click or drag & drop to upload stego carrier (PNG, 8-bit truecolor)</span>
                        </div>
                      )}
                      <input
                        type="file"
                        onChange={handleExtractCarrierUpload}
                        disabled={extracting}
                        accept="image/png"
                        className="absolute inset-0 opacity-0 cursor-pointer"
                      />
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-muted truncate">
                      File: <span className="text-[#ededed]">{extractFilename}</span>
                    </div>
                  </div>

                  {/* EXTRACTION CONFIGURATION & ACTION */}
                  <div className="border border-[#262626] rounded-lg p-4 bg-[#0d0d0d] flex flex-col justify-between">
                    <div>
                      <span className="text-xs font-mono font-semibold text-[#ededed] block mb-3 flex items-center">
                        <Key className="w-3.5 h-3.5 mr-1.5 text-primary" /> Cryptographic & Engine Settings
                      </span>

                      {/* AES-256 Key Input */}
                      <div className="space-y-1.5 mb-4">
                        <label className="text-[11px] font-mono text-muted block">
                          AES-256 Key (Hex String, Optional):
                        </label>
                        <input
                          type="text"
                          value={extractKeyHex}
                          onChange={(e) => setExtractKeyHex(e.target.value)}
                          disabled={extracting}
                          placeholder="Default standard 32-byte key if empty (64 hex chars)"
                          className="w-full bg-[#141414] border border-[#333] text-[#ededed] px-3 py-1.5 rounded text-xs font-mono focus:border-cyan-400 focus:outline-none placeholder:text-muted/40"
                        />
                        <span className="text-[10px] text-muted block font-mono">
                          Leave blank to use the standard Phase 10-13 symmetric key.
                        </span>
                      </div>

                      {/* Engine & Precision Summary */}
                      <div className="p-2.5 bg-[#141414] border border-[#262626] rounded font-mono text-xs space-y-1.5 mb-4">
                        <div className="flex justify-between">
                          <span className="text-muted">Modulation Engine:</span>
                          <span className="text-[#ededed] font-semibold">{selectedModel}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted">Model Precision:</span>
                          <span className="text-emerald-400 font-semibold uppercase">{selectedPrecision}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted">Security Policy:</span>
                          <span className="text-cyan-400">Fail-Closed Authentication</span>
                        </div>
                      </div>
                    </div>

                    {/* Extraction Action Button */}
                    <div className="pt-3 border-t border-[#262626]">
                      <button
                        onClick={handleStandaloneExtract}
                        disabled={extracting || !extractFile}
                        className="w-full py-2.5 px-4 text-xs font-mono font-semibold bg-cyan-600 hover:bg-cyan-500 text-white rounded disabled:opacity-40 flex items-center justify-center shadow-sm transition-all"
                      >
                        {extracting ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            EXTRACTING SECRET...
                          </>
                        ) : (
                          <>
                            <Unlock className="w-4 h-4 mr-2" />
                            EXTRACT HIDDEN SECRET PAYLOAD
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* EXTRACTION RESULT / RECOVERED PAYLOAD VIEW */}
              <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold tracking-wide flex items-center text-muted uppercase font-mono">
                    <FileText className="w-4 h-4 mr-2 text-primary" />
                    Extraction Result & Reconstructed Payload
                  </h3>
                  {extractResult && (
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">
                      STATUS: SUCCESSFUL
                    </span>
                  )}
                  {extractError && (
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
                      STATUS: FAILED CLOSED
                    </span>
                  )}
                </div>

                {extractResult ? (
                  <div className="space-y-4 font-mono text-xs">
                    <div className="p-3 bg-green-500/10 border border-green-500/30 rounded flex items-center space-x-2 text-green-400">
                      <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                      <span>
                        Payload successfully extracted, Reed-Solomon decoded, and AES-256-GCM authentication tag verified.
                      </span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                        <span className="text-[10px] text-muted block uppercase">Protocol Version</span>
                        <span className="text-sm font-bold text-cyan-400">v{extractResult.wire_protocol_version}</span>
                      </div>
                      <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                        <span className="text-[10px] text-muted block uppercase">ECC Parity</span>
                        <span className="text-sm font-bold text-[#ededed]">RS 2t={extractResult.ecc_parity}</span>
                      </div>
                      <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                        <span className="text-[10px] text-muted block uppercase">Session Counter</span>
                        <span className="text-sm font-bold text-[#ededed]">{extractResult.session_counter}</span>
                      </div>
                      <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                        <span className="text-[10px] text-muted block uppercase">Recovered Bytes</span>
                        <span className="text-sm font-bold text-green-400">{extractResult.recovered_bytes} B</span>
                      </div>
                    </div>

                    {extractResult.pqc_authenticated_sender_id && (
                      <div className="p-2.5 bg-[#0a0a0a] border border-primary/30 rounded text-xs text-primary flex items-center justify-between">
                        <span>PQC Authenticated Sender Identity:</span>
                        <span className="font-bold">{extractResult.pqc_authenticated_sender_id}</span>
                      </div>
                    )}

                    {extractedMediaUrl && (
                      <div className="p-4 bg-[#0a0a0a] border border-[#262626] rounded flex flex-col items-center">
                        <span className="text-xs text-muted mb-2">Recovered Media Preview:</span>
                        <img
                          src={extractedMediaUrl}
                          alt="Extracted Secret"
                          className="max-h-56 max-w-full object-contain rounded border border-[#333] mb-3"
                        />
                        <button
                          onClick={() => setIsSecretRecoveryOpen(true)}
                          className="text-xs text-primary hover:text-primary/80 font-semibold underline flex items-center"
                        >
                          Open in Full Modal Inspector
                        </button>
                      </div>
                    )}
                  </div>
                ) : extractError ? (
                  <div className="p-4 bg-red-500/10 border border-red-500/30 rounded font-mono text-xs text-red-400 space-y-2">
                    <div className="flex items-center space-x-2 font-bold text-red-300">
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                      <span>EXTRACTION FAILED CLOSED</span>
                    </div>
                    <p className="text-[#ededed]">{extractError}</p>
                    <p className="text-[11px] text-muted">
                      Security Invariant: When carrier headers are damaged, Reed-Solomon uncorrectable error thresholds are exceeded, or AES-GCM GHASH tags do not authenticate, PaladinMedia aborts cleanly without returning unverified plaintext. Ordinary (non-stego) photos will be rejected because they lack the PaladinMedia carrier header.
                    </p>
                  </div>
                ) : (
                  <div className="p-6 text-center text-muted font-mono text-xs">
                    <Unlock className="w-8 h-8 opacity-30 mx-auto mb-2" />
                    <span>Upload a stego carrier PNG, click &quot;Load Sample Stego&quot;, or transfer the stego container from an Experiment run to extract the hidden payload.</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* RIGHT COLUMN: REAL PIPELINE TIMELINE & EXECUTION LOG */}
        <div className="flex flex-col space-y-6 overflow-hidden">
          
          {/* PIPELINE / EXTRACTION LIFECYCLE */}
          <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm flex-1 overflow-y-auto">
            <h2 className="text-sm font-semibold tracking-wide text-muted uppercase font-mono mb-4 flex items-center">
              <Zap className="w-4 h-4 mr-2 text-primary" />
              {activeTab === "EXPERIMENT" ? "Transform-Domain Pipeline Stages" : "Extraction Lifecycle Verification"}
            </h2>
            {activeTab === "EXPERIMENT" ? (
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
                          <span className={`font-semibold ${textCol}`}>
                            {idx + 1}. {
                              s === "EMBED"
                                ? (selectedModel === "parity_classical" ? "Deterministic Parity Embedding" : "ACS-Sub-INN Neural Modulation")
                                : s === "EXTRACT"
                                ? (selectedModel === "parity_classical" ? "Header & Payload Extraction" : "ACS-Sub-INN Neural Extraction")
                                : STAGE_LABELS[s]
                            }
                          </span>
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
            ) : (
              <div className="space-y-2 font-mono text-xs">
                {[
                  {
                    title: "1. Carrier Validation & PNG Signature",
                    desc: "Lossless 8-bit truecolor RGB, even dimensions (>=32x32, <=2048x2048)",
                    status: extractFile ? (extractError ? "failed" : extractResult ? "complete" : extracting ? "processing" : "waiting") : "waiting",
                  },
                  {
                    title: "2. 1-Level 2D Haar DWT",
                    desc: "Orthogonal decomposition into LL, LH, HL, and HH subbands",
                    status: extracting ? "processing" : extractResult ? "complete" : extractError ? "failed" : "waiting",
                  },
                  {
                    title: "3. Transform Demodulation",
                    desc: selectedModel === "parity_classical" ? "Classical parity demodulation" : "ACS-Sub-INN neural subband demodulation",
                    status: extracting ? "processing" : extractResult ? "complete" : extractError ? "failed" : "waiting",
                  },
                  {
                    title: "4. Reed-Solomon Error Correction",
                    desc: "GF(2^8) Reed-Solomon syndrome decode & bit error repair",
                    status: extracting ? "processing" : extractResult ? "complete" : extractError ? "failed" : "waiting",
                  },
                  {
                    title: "5. AES-256-GCM Tag Verification",
                    desc: "GHASH authentication tag verification (fail-closed integrity gate)",
                    status: extractResult ? "complete" : extractError ? "failed" : extracting ? "processing" : "waiting",
                  },
                  {
                    title: "6. Secret Media Reconstruction",
                    desc: "De-framing, CRC-32 integrity validation, and binary recovery",
                    status: extractResult ? "complete" : extractError ? "failed" : extracting ? "processing" : "waiting",
                  },
                ].map((stg, i) => {
                  let borderCol = "border-[#212121]";
                  let textCol = "text-muted";
                  let dotCol = "bg-[#333]";

                  if (stg.status === "processing") {
                    borderCol = "border-cyan-500/50 bg-cyan-500/5";
                    textCol = "text-cyan-400";
                    dotCol = "bg-cyan-400 animate-pulse";
                  } else if (stg.status === "complete") {
                    borderCol = "border-green-500/30 bg-green-500/5";
                    textCol = "text-green-400";
                    dotCol = "bg-green-500";
                  } else if (stg.status === "failed") {
                    borderCol = "border-red-500/40 bg-red-500/10";
                    textCol = "text-red-400";
                    dotCol = "bg-red-500";
                  }

                  return (
                    <div key={i} className={`p-2.5 rounded border ${borderCol} transition-colors flex items-start`}>
                      <div className="mt-1 mr-2.5 flex-shrink-0">
                        <div className={`w-2.5 h-2.5 rounded-full ${dotCol}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex justify-between items-center">
                          <span className={`font-semibold ${textCol}`}>{stg.title}</span>
                          <span className={`text-[10px] uppercase font-bold ${textCol}`}>{stg.status}</span>
                        </div>
                        <p className="text-[11px] text-muted mt-0.5">{stg.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
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

      {/* OPERATOR AUTHENTICATION MODAL (BFF / OIDC) */}
      {loginModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 font-mono">
          <div className="bg-[#121212] border border-[#333] rounded-lg max-w-md w-full p-6 shadow-2xl">
            <div className="flex items-center space-x-3 mb-4">
              <div className="p-2 bg-primary/20 rounded-md text-primary">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-[#ededed]">Enterprise Single Sign-On</h3>
                <p className="text-[11px] text-muted">OpenID Connect / OAuth 2.0 (PKCE) BFF Gate</p>
              </div>
            </div>

            <p className="text-xs text-muted mb-4 leading-relaxed">
              This PaladinMedia instance requires per-user authentication via corporate Identity Provider. Click below to sign in using OAuth 2.0 Authorization Code Flow with PKCE. A cryptographically secure, HttpOnly session cookie will be established.
            </p>

            <div className="bg-[#181818] border border-[#282828] rounded p-3 mb-4 text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-muted">Identity Provider:</span>
                <span className="text-[#ededed] font-semibold">{authSession.oidc_issuer || "Configured via PALADIN_OIDC_ISSUER"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">IdP Gate Status:</span>
                <span className="text-amber-400 font-semibold">{authSession.idp_gate_status || "OPEN_PENDING_EXTERNAL_IDP"}</span>
              </div>
            </div>

            {loginError && (
              <div className="p-2.5 bg-red-950/40 border border-red-800/60 rounded text-red-300 text-xs flex items-center space-x-2 mb-4">
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                <span>{loginError}</span>
              </div>
            )}

            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={handleOidcLogin}
                className="w-full px-4 py-2.5 bg-primary hover:bg-primary/90 text-black font-semibold rounded text-xs transition-colors flex items-center justify-center space-x-2 cursor-pointer shadow"
              >
                <Shield className="w-4 h-4" />
                <span>Sign In with Identity Provider (SSO)</span>
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
