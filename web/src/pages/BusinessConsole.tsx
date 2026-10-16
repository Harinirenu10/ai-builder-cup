import React from 'react';
import { ShieldCheck, TrendingDown, CheckCircle, AlertTriangle, Database } from 'lucide-react';

export const BusinessConsole: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-slate-900 to-indigo-950 text-white border border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-black">Merchant &amp; Business Console</h1>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold uppercase">
              B2B Partner API
            </span>
          </div>
          <p className="text-sm text-slate-300 mt-1">
            Real-time fit analytics, size curve diagnostics, and return reduction intelligence.
          </p>
        </div>

        {/* Privacy Enforcement Notice */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-800/90 border border-slate-700 text-xs text-slate-200">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Verdict-Only API: Customer body measurements are never exposed</span>
        </div>
      </div>

      {/* Simulated Data Marker */}
      <div className="p-3.5 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/60 flex items-center space-x-3 text-xs text-amber-800 dark:text-amber-300">
        <Database className="w-4 h-4 flex-shrink-0" />
        <span><strong>Simulated Data:</strong> The analytics below are simulated for prototype evaluation according to project rules.</span>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Estimated Return Drop</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-black text-emerald-600 dark:text-emerald-400">-34.8%</span>
            <TrendingDown className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-xs text-slate-500 mt-1">Across top 12 return-heavy SKUs</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Fit Recommendation Match</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-black text-sky-600 dark:text-sky-400">92.4%</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Shoppers kept first ordered size</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Garment Scans Calibrated</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-black text-indigo-600 dark:text-indigo-400">1,480</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">A4 flat-lay reference verified</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Active Twins Connected</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-black text-purple-600 dark:text-purple-400">8,240</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Consumer 3D profiles active</p>
        </div>
      </div>

      {/* Sizing Diagnostic Table */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white">Product Sizing Anomaly Diagnostics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                <th className="pb-3">SKU / Garment</th>
                <th className="pb-3">Stated Alpha Size</th>
                <th className="pb-3">True Flat-Lay Chest</th>
                <th className="pb-3">Brand Chart Spec</th>
                <th className="pb-3">Diagnosis</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              <tr>
                <td className="py-3 font-semibold text-slate-900 dark:text-white">DTC-TEE-WHT-M</td>
                <td className="py-3">M</td>
                <td className="py-3 font-mono">104.2 cm</td>
                <td className="py-3 font-mono">104.0 cm</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">Conforms to pattern ease (&plusmn;0.2 cm)</td>
                <td className="py-3">
                  <span className="inline-flex items-center space-x-1 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                    <CheckCircle className="w-3.5 h-3.5" />
                    <span>Exact Fit</span>
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 font-semibold text-slate-900 dark:text-white">DTC-JKT-DNM-L</td>
                <td className="py-3">L</td>
                <td className="py-3 font-mono">110.5 cm</td>
                <td className="py-3 font-mono">116.0 cm</td>
                <td className="py-3 text-amber-600 dark:text-amber-400">Garment runs 5.5 cm smaller than published chart</td>
                <td className="py-3">
                  <span className="inline-flex items-center space-x-1 text-xs text-amber-600 dark:text-amber-400 font-medium">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Chart Drift</span>
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 font-semibold text-slate-900 dark:text-white">DTC-TRZ-SLM-32</td>
                <td className="py-3">32</td>
                <td className="py-3 font-mono">83.8 cm</td>
                <td className="py-3 font-mono">84.0 cm</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">Waistband rise aligns with low-rise specs</td>
                <td className="py-3">
                  <span className="inline-flex items-center space-x-1 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                    <CheckCircle className="w-3.5 h-3.5" />
                    <span>Exact Fit</span>
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
