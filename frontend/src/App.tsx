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
  Layers, 
  Zap
} from 'lucide-react';

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

export default function App() {
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<"IDLE" | "PROCESSING" | "SUCCESS" | "FAILED">("IDLE");
  const [stages, setStages] = useState<Record<PipelineStage, StageState>>(
    STAGES.reduce((acc, s) => ({ ...acc, [s]: { status: "waiting" } }), {} as any)
  );
  const [logs, setLogs] = useState<string[]>([]);

  // Media state
  const [secretSessionId, setSecretSessionId] = useState<string | null>(null);
  const [secretPreview, setSecretPreview] = useState<string | null>(null);
  const [secretFilename, setSecretFilename] = useState<string>("No file chosen");
  const [secretBytesSize, setSecretBytesSize] = useState<number>(0);

  const [coverSessionId, setCoverSessionId] = useState<string | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const [coverFilename, setCoverFilename] = useState<string>("No file chosen");

  // Output artifacts
  const [stegoImage, setStegoImage] = useState<string | null>(null);
  const [recoveredImage, setRecoveredImage] = useState<string | null>(null);
  const [dwtSubbands, setDwtSubbands] = useState<Record<string, string> | null>(null);
  const [modifiedHhImage, setModifiedHhImage] = useState<string | null>(null);
  const [psnr, setPsnr] = useState<number | null>(null);
  const [ssim, setSsim] = useState<number | null>(null);
  const [exactMatch, setExactMatch] = useState<boolean | null>(null);

  // Capacity metadata
  const [capacityInfo, setCapacityInfo] = useState<any>(null);

  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Automatically load default verified fixtures on start
    backend.fetchDefaultFixtures()
      .then((fixtures: DefaultFixtures) => {
        if (fixtures.secret && fixtures.secret.data_url) {
          setSecretSessionId(fixtures.secret.session_id);
          setSecretPreview(fixtures.secret.data_url);
          setSecretFilename(fixtures.secret.name);
          setSecretBytesSize(fixtures.secret.size);
        }
        if (fixtures.cover && fixtures.cover.data_url) {
          setCoverSessionId(fixtures.cover.session_id);
          setCoverPreview(fixtures.cover.data_url);
          setCoverFilename(fixtures.cover.name);
        }
        addLog("Preloaded baseline fixtures: secret_small.png (24x24) & 0001x2.png (512x512)");
      })
      .catch((err) => {
        console.warn("Could not load default fixtures automatically:", err);
      });
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
      setSecretPreview(URL.createObjectURL(file));
      addLog(`Selected secret media file: ${file.name} (${file.size} bytes)`);

      try {
        const uploadRes = await backend.uploadMedia(file);
        setSecretSessionId(uploadRes.session_id);
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
      setCoverPreview(URL.createObjectURL(file));
      addLog(`Selected cover carrier image: ${file.name} (${file.size} bytes)`);

      try {
        const uploadRes = await backend.uploadMedia(file);
        setCoverSessionId(uploadRes.session_id);
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
    setStages(STAGES.reduce((acc, s) => ({ ...acc, [s]: { status: "waiting" } }), {} as any));
  };

  const handleRun = () => {
    if (!secretSessionId) {
      addLog("ERROR: Please select or upload secret media first.");
      return;
    }

    setRunning(true);
    setRunStatus("PROCESSING");
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
        if (event.stage === "CAPACITY_CHECK" && event.metadata) {
          setCapacityInfo(event.metadata);
        }
        if (event.stage === "DWT" && event.metadata?.subbands) {
          setDwtSubbands(event.metadata.subbands);
        }
        if (event.stage === "EMBED" && event.metadata?.modified_hh) {
          setModifiedHhImage(event.metadata.modified_hh);
        }
        if (event.stage === "IDWT" && event.metadata) {
          if (event.metadata.stego_image) setStegoImage(event.metadata.stego_image);
          if (event.metadata.psnr_db) setPsnr(event.metadata.psnr_db);
          if (event.metadata.ssim) setSsim(event.metadata.ssim);
        }
        if (event.stage === "SECRET_RECOVER" && event.metadata?.recovered_image) {
          setRecoveredImage(event.metadata.recovered_image);
        }
        if (event.stage === "EXACT_MATCH_VERIFY" && event.metadata) {
          setExactMatch(Boolean(event.metadata.exact_match));
        }

        if (event.status === "failed") {
          setRunStatus("FAILED");
          addLog(`PIPELINE TERMINATED: ${event.error || "Stage execution failed"}`);
        }
      },
      () => {
        setRunning(false);
        setStages(prev => {
          const isVerified = prev["EXACT_MATCH_VERIFY"].status === "complete";
          const isFailed = Object.values(prev).some(s => s.status === "failed");
          if (isVerified) {
            setRunStatus("SUCCESS");
          } else if (isFailed) {
            setRunStatus("FAILED");
          } else {
            setRunStatus("IDLE");
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
    setRunStatus("IDLE");
    addLog("Run cancelled by operator.");
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
            <span className="text-muted text-[10px] uppercase">Dataset</span>
            <span className="text-[#ededed] font-medium">MVP 0 Dataset v0.1 (512x512)</span>
          </div>
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Backend Host</span>
            <span className="text-green-500 font-medium flex items-center">
              <span className="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></span>
              PORT 8000 CONNECTED
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-muted text-[10px] uppercase">Run Status</span>
            <span className={`font-semibold uppercase px-2 py-0.5 rounded text-center ${
              runStatus === "SUCCESS" ? "bg-green-500/20 text-green-400 border border-green-500/30" :
              runStatus === "PROCESSING" ? "bg-primary/20 text-primary border border-primary/30 animate-pulse" :
              runStatus === "FAILED" ? "bg-red-500/20 text-red-400 border border-red-500/30" :
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
                  onClick={() => {
                    backend.fetchDefaultFixtures().then(fixtures => {
                      setSecretSessionId(fixtures.secret.session_id);
                      setSecretPreview(fixtures.secret.data_url);
                      setSecretFilename(fixtures.secret.name);
                      setSecretBytesSize(fixtures.secret.size);
                      addLog("Loaded verified baseline test fixture (secret_small.png)");
                    });
                  }}
                  className="text-[11px] bg-[#1a1a1a] hover:bg-[#262626] border border-[#333] px-2.5 py-1 rounded font-mono text-muted hover:text-[#ededed] transition-colors"
                >
                  Load Small Fixture (Fits)
                </button>
              </div>
            </div>
            
            {/* 4 CARDS: SECRET, COVER, STEGO, RECOVERED */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* CARD 1: SECRET MEDIA */}
              <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col relative group">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-semibold text-primary">1. SECRET MEDIA (PAYLOAD)</span>
                  <span className="text-[10px] font-mono text-muted bg-[#1c1c1c] px-1.5 py-0.5 rounded">
                    {secretBytesSize > 0 ? `${secretBytesSize} B` : "No media"}
                  </span>
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
                    512×512 RGB
                  </span>
                </div>
                <div className="h-44 border border-dashed border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {coverPreview ? (
                    <img src={coverPreview} alt="Cover" className="max-h-full max-w-full object-contain p-2" />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="w-6 h-6 text-muted mx-auto mb-1 group-hover:text-[#888] transition-colors" />
                      <span className="text-xs text-muted block">Click to upload 512x512 carrier</span>
                    </div>
                  )}
                  <input type="file" onChange={handleCoverUpload} disabled={running} accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" />
                </div>
                <div className="mt-2 text-[11px] font-mono text-muted truncate">
                  Carrier: <span className="text-[#ededed]">{coverFilename}</span>
                </div>
              </div>

              {/* CARD 3: STEGO MEDIA */}
              <div className="border border-[#262626] rounded-lg p-3 bg-[#0d0d0d] flex flex-col">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-semibold text-green-400">3. STEGO MEDIA (RECONSTRUCTED)</span>
                  {psnr !== null && (
                    <span className="text-[10px] font-mono text-green-400 bg-green-500/10 border border-green-500/20 px-1.5 py-0.5 rounded">
                      PSNR: {psnr} dB
                    </span>
                  )}
                </div>
                <div className="h-44 border border-[#2b2b2b] rounded flex flex-col items-center justify-center relative overflow-hidden bg-[#070707]">
                  {stegoImage ? (
                    <img src={stegoImage} alt="Stego" className="max-h-full max-w-full object-contain p-2" />
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
                  ) : (
                    <div className="text-center p-4 text-muted">
                      <Zap className="w-6 h-6 mx-auto mb-1 opacity-40" />
                      <span className="text-xs block">Awaiting receiver recovery</span>
                    </div>
                  )}
                </div>
                <div className="mt-2 text-[11px] font-mono text-muted">
                  Status: <span className={exactMatch ? "text-green-400 font-semibold" : "text-[#ededed]"}>
                    {exactMatch === true ? "VERBATIM EQUALITY VERIFIED" : exactMatch === false ? "VERIFICATION FAILED" : "STANDBY"}
                  </span>
                </div>
              </div>

            </div>

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

          {/* REAL DWT SUBBAND VISUALIZER */}
          <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-sm font-semibold tracking-wide flex items-center text-muted uppercase font-mono">
                <Layers className="w-4 h-4 mr-2 text-primary" />
                DWT Subband Transform Inspection (Actual Coefficients)
              </h2>
              <span className="text-[10px] font-mono text-muted">1-Level 2D Haar (256x256 per band)</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {["LL", "LH", "HL", "HH"].map((band) => (
                <div key={band} className="border border-[#262626] rounded bg-[#0a0a0a] p-2 flex flex-col items-center">
                  <span className="text-[11px] font-mono font-semibold mb-1 text-muted">Subband: {band}</span>
                  <div className="w-full aspect-square bg-[#050505] rounded border border-[#1f1f1f] flex items-center justify-center overflow-hidden">
                    {dwtSubbands && dwtSubbands[band] ? (
                      <img src={dwtSubbands[band]} alt={band} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-[10px] font-mono text-[#333]">NOT LOADED</span>
                    )}
                  </div>
                </div>
              ))}

              {/* MODIFIED HH' */}
              <div className="border border-primary/30 rounded bg-[#0a0a0a] p-2 flex flex-col items-center">
                <span className="text-[11px] font-mono font-semibold mb-1 text-primary">Modified HH' (Stego)</span>
                <div className="w-full aspect-square bg-[#050505] rounded border border-primary/20 flex items-center justify-center overflow-hidden">
                  {modifiedHhImage ? (
                    <img src={modifiedHhImage} alt="Modified HH" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-[10px] font-mono text-[#333]">NOT EMBEDDED</span>
                  )}
                </div>
              </div>
            </div>
          </div>

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
                <span className="text-sm font-bold text-[#ededed]">8,192 B</span>
                <span className="text-[9px] text-muted block mt-0.5">256×256 bits = 1 HH channel</span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Max Protected</span>
                <span className="text-sm font-bold text-[#ededed]">8,176 B</span>
                <span className="text-[9px] text-muted block mt-0.5">8,192 B - 16B Carrier Header</span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Usable Plaintext</span>
                <span className="text-sm font-bold text-primary">~7,088 B</span>
                <span className="text-[9px] text-muted block mt-0.5">After RS parity + AES + Wire</span>
              </div>
              <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded">
                <span className="text-[10px] text-muted block uppercase">Protected Payload</span>
                <span className={`text-sm font-bold ${
                  capacityInfo && !capacityInfo.fits ? "text-red-400" : "text-[#ededed]"
                }`}>
                  {capacityInfo ? `${capacityInfo.protected_payload_bytes} B` : "-- B"}
                </span>
                <span className="text-[9px] text-muted block mt-0.5">
                  {capacityInfo ? `Remaining: ${capacityInfo.remaining_capacity_bytes} B` : "Awaiting check"}
                </span>
              </div>
            </div>

            {/* Capacity Visual Progress Bar */}
            <div className="w-full bg-[#1c1c1c] h-2 rounded-full overflow-hidden border border-[#2b2b2b]">
              <div 
                className={`h-full transition-all duration-500 ${
                  capacityInfo && !capacityInfo.fits ? "bg-red-500" : "bg-primary"
                }`}
                style={{
                  width: capacityInfo 
                    ? `${Math.min(100, (capacityInfo.protected_payload_bytes / capacityInfo.max_protected_bytes) * 100)}%` 
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
    </div>
  );
}
