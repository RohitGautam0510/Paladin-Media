import React from 'react';
import { Leaf, Cpu, Zap, Activity, HardDrive, CheckCircle2, AlertCircle } from 'lucide-react';
import type { SustainabilityMetrics } from '../services/backendAdapter';

interface SustainabilityWidgetProps {
  telemetry: SustainabilityMetrics | null;
  isRunning?: boolean;
}

export const SustainabilityWidget: React.FC<SustainabilityWidgetProps> = ({
  telemetry,
  isRunning = false,
}) => {
  return (
    <div className="bg-[#121212] border border-[#262626] rounded-xl p-5 shadow-sm font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide flex items-center text-muted uppercase">
          <Leaf className="w-4 h-4 mr-2 text-green-400" />
          Green Computing & Sustainability Telemetry
        </h2>

        {telemetry ? (
          <div className="flex items-center space-x-2">
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded flex items-center ${
                telemetry.is_estimated
                  ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                  : "bg-green-500/15 text-green-300 border border-green-500/30"
              }`}
            >
              {telemetry.is_estimated ? (
                <>
                  <AlertCircle className="w-3 h-3 mr-1" />
                  HARDWARE-PROFILED (ESTIMATED)
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  CODECARBON DIRECT SENSING
                </>
              )}
            </span>
          </div>
        ) : isRunning ? (
          <span className="text-[10px] bg-primary/10 text-primary border border-primary/20 px-2 py-0.5 rounded animate-pulse">
            SAMPLING POWER & CPU UTILIZATION...
          </span>
        ) : (
          <span className="text-[10px] bg-[#1a1a1a] text-muted border border-[#333] px-2 py-0.5 rounded">
            TELEMETRY STANDBY
          </span>
        )}
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-3">
        {/* Metric 1: Execution Latency */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <Activity className="w-3 h-3 mr-1 text-primary" />
            Execution Latency
          </div>
          <span className="text-sm font-bold text-[#ededed]">
            {telemetry ? `${telemetry.execution_time_ms} ms` : isRunning ? "Profiling..." : "-- ms"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">Wall-clock inference</span>
        </div>

        {/* Metric 2: CPU Utilization */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <Cpu className="w-3 h-3 mr-1 text-cyan-400" />
            CPU Utilization
          </div>
          <span className="text-sm font-bold text-cyan-400">
            {telemetry ? `${telemetry.cpu_utilization_percent}%` : isRunning ? "Active" : "-- %"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">Process core load</span>
        </div>

        {/* Metric 3: Resident Memory */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <HardDrive className="w-3 h-3 mr-1 text-purple-400" />
            Memory RSS
          </div>
          <span className="text-sm font-bold text-[#ededed]">
            {telemetry ? `${telemetry.memory_rss_mb} MB` : isRunning ? "Active" : "-- MB"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">
            {telemetry
              ? `Delta: ${telemetry.memory_delta_mb >= 0 ? `+${telemetry.memory_delta_mb}` : telemetry.memory_delta_mb} MB`
              : "Peak allocation"}
          </span>
        </div>

        {/* Metric 4: Estimated Power */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <Zap className="w-3 h-3 mr-1 text-amber-400" />
            Average Power
          </div>
          <span className="text-sm font-bold text-amber-300">
            {telemetry ? `${telemetry.estimated_power_watts} W` : isRunning ? "Sampling..." : "-- W"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">CPU TDP + RAM scaling</span>
        </div>

        {/* Metric 5: Energy Consumption */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <Leaf className="w-3 h-3 mr-1 text-green-400" />
            Energy / Roundtrip
          </div>
          <span className="text-sm font-bold text-green-400">
            {telemetry ? `${(telemetry.energy_kwh * 1000).toFixed(4)} mWh` : isRunning ? "Computing..." : "-- mWh"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">
            {telemetry ? `${telemetry.energy_kwh.toExponential(2)} kWh` : "Joules per container"}
          </span>
        </div>

        {/* Metric 6: Carbon Emissions */}
        <div className="bg-[#0a0a0a] border border-[#262626] p-2.5 rounded flex flex-col justify-between">
          <div className="flex items-center text-[10px] text-muted uppercase mb-1">
            <Leaf className="w-3 h-3 mr-1 text-emerald-300" />
            CO₂ Emissions
          </div>
          <span className="text-sm font-bold text-emerald-300">
            {telemetry ? `${(telemetry.co2_emissions_grams * 1000).toFixed(2)} mg` : isRunning ? "Computing..." : "-- mg"}
          </span>
          <span className="text-[9px] text-muted mt-0.5">
            {telemetry ? `${telemetry.co2_emissions_grams.toFixed(5)} g CO₂` : "Grid: 475 g/kWh"}
          </span>
        </div>
      </div>

      {/* Footer / Transparency Disclaimer */}
      <div className="pt-2 text-[10px] text-muted flex flex-col sm:flex-row justify-between border-t border-[#1c1c1c] gap-1">
        <span>
          <strong>Methodology:</strong> {telemetry?.telemetry_source === "codecarbon"
            ? "Direct RAPL / GPU / Intel Power Gadget sensing via CodeCarbon"
            : "Thermal Design Power (TDP) dynamic CPU frequency & memory density estimation model (no fake zeros)."}
        </span>
        <span className="text-right">
          Grid Intensity: <span className="text-[#ededed]">475 g CO₂-eq/kWh</span> | Free Cloud Tier Optimised
        </span>
      </div>
    </div>
  );
};
