import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, UserCheck, Ruler, Sparkles, Shield, BarChart3, Store } from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="relative overflow-hidden">
      {/* Background radial gradients */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-gradient-to-tr from-sky-400/20 via-indigo-400/20 to-purple-400/20 blur-3xl pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-24">
        {/* Hero Section */}
        <div className="text-center max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-sky-100/80 dark:bg-sky-950/80 border border-sky-300 dark:border-sky-800 text-sky-700 dark:text-sky-300 text-xs font-semibold tracking-wide">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Google Cloud AI Builder Cup &bull; Retail &amp; Commerce</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-slate-900 dark:text-white leading-[1.1]">
            Measured,{' '}
            <span className="bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-600 bg-clip-text text-transparent">
              not guessed.
            </span>
          </h1>

          <p className="text-lg sm:text-xl text-slate-600 dark:text-slate-300 font-normal leading-relaxed">
            We measure the body (a 3D twin from 4 photos) and the garment (one flat-lay photo next to an A4 sheet) so shoppers and businesses can trust size and fit answers.
          </p>

          {/* Call to action portals */}
          <div className="pt-4 flex flex-wrap justify-center gap-4">
            <Link
              to="/app"
              className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold text-base shadow-lg shadow-sky-600/25 transition-all transform hover:-translate-y-0.5"
            >
              <span>Launch Customer PWA</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/biz"
              className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-slate-900 dark:bg-slate-800 hover:bg-slate-800 dark:hover:bg-slate-700 text-white font-semibold text-base transition-all border border-slate-700"
            >
              <BarChart3 className="w-4 h-4" />
              <span>Business Console</span>
            </Link>
            <Link
              to="/store"
              className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 font-semibold text-base transition-all border border-slate-300 dark:border-slate-700 shadow-sm"
            >
              <Store className="w-4 h-4 text-sky-500" />
              <span>Demo Threads Co.</span>
            </Link>
          </div>
        </div>

        {/* Feature Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20">
          <div className="glass-panel p-6 rounded-2xl shadow-sm border border-slate-200/60 dark:border-slate-800 space-y-4">
            <div className="h-12 w-12 rounded-xl bg-sky-100 dark:bg-sky-950 text-sky-600 dark:text-sky-400 flex items-center justify-center">
              <UserCheck className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white">4-Photo 3D Body Twin</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              MediaPipe Pose landmarking combined with parametric SMPL fitting extracts accurate millimeter-level body measurements without sensitive cloud photo storage.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl shadow-sm border border-slate-200/60 dark:border-slate-800 space-y-4">
            <div className="h-12 w-12 rounded-xl bg-indigo-100 dark:indigo-950 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
              <Ruler className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white">Garment Flat-Lay A4 Calibration</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              Place any garment next to a standard A4 sheet to calibrate pixels into physical centimeters. Deterministic geometry extracts seams, chest, waist, and length.
            </p>
          </div>

          <div className="glass-panel p-6 rounded-2xl shadow-sm border border-slate-200/60 dark:border-slate-800 space-y-4">
            <div className="h-12 w-12 rounded-xl bg-emerald-100 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
              <Shield className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white">Verdict-Only Business Privacy</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              Business APIs return fit verdicts and ease ratings—never raw body measurements. Users own and control their data, and 18+ verification is mandatory.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
