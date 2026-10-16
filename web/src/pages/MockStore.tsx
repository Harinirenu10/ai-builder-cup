import React, { useState } from 'react';
import { ShoppingBag, Sparkles, Check, Info, ShieldCheck, Ruler } from 'lucide-react';

interface MockProduct {
  id: string;
  title: string;
  price: string;
  material: string;
  sizes: string[];
  recommendedSize: string;
  verdict: string;
  easeNotes: string;
}

export const MockStore: React.FC = () => {
  const [selectedProduct, setSelectedProduct] = useState<number>(0);
  const [selectedSize, setSelectedSize] = useState<string>('M');
  const [twinApplied, setTwinApplied] = useState<boolean>(true);

  const products: MockProduct[] = [
    {
      id: 'dtc-001',
      title: 'Heavyweight Boxy Crewneck Tee',
      price: '$48.00',
      material: '100% Organic Heavyweight Cotton (240 GSM)',
      sizes: ['S', 'M', 'L', 'XL'],
      recommendedSize: 'M',
      verdict: 'Comfortable relaxed ease (+5.2 cm room at chest). Sleeves drop cleanly off the shoulder.',
      easeNotes: 'Flat-lay measurement: chest 104.0 cm, length 72.0 cm. Designed with a structured drape and zero synthetic stretch.',
    },
    {
      id: 'dtc-002',
      title: 'Relaxed Taper Selvedge Denim',
      price: '$120.00',
      material: '98% Cotton, 2% Elastane (13.5 oz Japanese Denim)',
      sizes: ['30', '32', '34', '36'],
      recommendedSize: '32',
      verdict: 'Clean tailored waist (+1.5 cm ease) with generous thigh circumference (+3.8 cm ease).',
      easeNotes: 'Flat-lay measurement: waist 84.0 cm, inseam 81.0 cm. Fabric eases 1-2 cm after first 3 wears.',
    },
    {
      id: 'dtc-003',
      title: 'Field Canvas Utility Overshirt',
      price: '$95.00',
      material: '100% Cotton Duck Canvas',
      sizes: ['S', 'M', 'L', 'XL'],
      recommendedSize: 'L',
      verdict: 'Generous layering silhouette (+8.0 cm chest ease) allowing hoodies or knitwear beneath.',
      easeNotes: 'Flat-lay measurement: chest 112.0 cm, sleeve length 66.0 cm. Unlined rigid weave.',
    },
  ];

  const current = products[selectedProduct];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Brand Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-sky-950 to-slate-900 text-white shadow-xl">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-black tracking-tight">Demo Threads Co.</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-400/30 uppercase font-semibold">
              Mock Storefront
            </span>
          </div>
          <p className="text-sm text-slate-300 mt-1">
            Fictional ecommerce storefront with instant 3D twin fit integration.
          </p>
        </div>

        <div className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs">
          <Info className="w-4 h-4 text-sky-400" />
          <span><strong>Simulated Merchant Storefront</strong> for Hackathon Demo</span>
        </div>
      </div>

      {/* Main product showcase */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Product Selector & Preview */}
        <div className="space-y-4">
          <div className="glass-panel p-8 rounded-2xl border border-slate-200 dark:border-slate-800 flex flex-col items-center justify-center min-h-[380px] bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950 relative overflow-hidden">
            <div className="w-48 h-56 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-lg flex flex-col items-center justify-center p-4 text-center space-y-3">
              <ShoppingBag className="w-16 h-16 text-sky-500" />
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                {current.id}
              </div>
              <div className="text-sm font-bold text-slate-900 dark:text-white">
                {current.title}
              </div>
            </div>

            <div className="mt-6 flex space-x-3">
              {products.map((p, idx) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setSelectedProduct(idx);
                    setSelectedSize(p.recommendedSize);
                  }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    selectedProduct === idx
                      ? 'bg-sky-600 text-white shadow'
                      : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                  }`}
                >
                  {p.title.split(' ')[0]}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Product Details & Measured Fit Integration */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-6">
          <div>
            <div className="flex justify-between items-start">
              <h1 className="text-2xl font-black text-slate-900 dark:text-white">{current.title}</h1>
              <span className="text-2xl font-black text-sky-600 dark:text-sky-400">{current.price}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{current.material}</p>
          </div>

          {/* Measured Fit Recommendation Box */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950/40 border border-sky-200 dark:border-sky-800/60 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold text-sky-800 dark:text-sky-300">
                <Sparkles className="w-4 h-4 text-sky-500" />
                <span>Your Measured Twin Recommendation</span>
              </div>
              <button
                onClick={() => setTwinApplied(!twinApplied)}
                className="text-[11px] underline text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
              >
                {twinApplied ? 'Disconnect Twin' : 'Connect Twin'}
              </button>
            </div>

            {twinApplied ? (
              <div className="space-y-2">
                <div className="flex items-baseline space-x-2">
                  <span className="text-xs text-slate-600 dark:text-slate-400">Recommended Size:</span>
                  <span className="text-xl font-black text-sky-600 dark:text-sky-400">{current.recommendedSize}</span>
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold">(96% fit confidence)</span>
                </div>
                <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                  {current.verdict}
                </p>
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  {current.easeNotes}
                </p>
              </div>
            ) : (
              <p className="text-xs text-slate-500">Connect your 3D twin to view verified fit verdicts.</p>
            )}
          </div>

          {/* Size Selector */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-900 dark:text-white uppercase tracking-wider">Select Size</span>
              <span className="text-slate-500 flex items-center space-x-1">
                <Ruler className="w-3.5 h-3.5" />
                <span>Garment Dimensions Verified</span>
              </span>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {current.sizes.map((sz) => {
                const isRec = sz === current.recommendedSize && twinApplied;
                const isSel = sz === selectedSize;
                return (
                  <button
                    key={sz}
                    onClick={() => setSelectedSize(sz)}
                    className={`py-3 rounded-xl font-bold text-sm transition-all relative ${
                      isSel
                        ? 'bg-sky-600 text-white shadow-md shadow-sky-600/30'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    <span>{sz}</span>
                    {isRec && (
                      <span className="absolute -top-2 -right-2 h-4 w-4 bg-emerald-500 text-white rounded-full flex items-center justify-center text-[10px] shadow">
                        <Check className="w-3 h-3" />
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Add to Cart button */}
          <button className="w-full py-4 rounded-xl bg-slate-900 dark:bg-sky-600 hover:bg-slate-800 dark:hover:bg-sky-500 text-white font-bold text-sm shadow-lg transition-transform transform active:scale-98">
            Add to Bag — {selectedSize}
          </button>

          {/* Privacy Footnote */}
          <div className="pt-2 flex items-center space-x-2 text-[11px] text-slate-500">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
            <span>Store receives fit verdict only; your body measurements are private to your twin.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
