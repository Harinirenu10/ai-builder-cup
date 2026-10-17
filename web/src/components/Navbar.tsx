import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sun, Moon, Sparkles, User, Briefcase, ShoppingBag, ShieldCheck } from 'lucide-react';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const [isDark, setIsDark] = useState(() => {
    return document.documentElement.classList.contains('dark') ||
      window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);

  const navLinks: { to: string; label: string; icon: React.ElementType; badge: string; external?: string }[] = [
    { to: '/app', label: 'Customer PWA', icon: User, badge: 'Consumer' },
    { to: '/biz', label: 'Business Console', icon: Briefcase, badge: 'Merchant' },
    { to: '/store', label: 'Demo Threads Co.', icon: ShoppingBag, badge: 'Storefront', external: '/store/' },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-200/80 dark:border-slate-800/80 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-3 group">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-500 via-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-md shadow-sky-500/20 group-hover:scale-105 transition-transform duration-200">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-slate-900 via-sky-900 to-indigo-900 dark:from-white dark:via-sky-200 dark:to-indigo-300 bg-clip-text text-transparent">
                Measured Fit
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300 font-semibold uppercase tracking-wider border border-sky-200/50 dark:border-sky-800/50">
                AI Builder Cup
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">Measured, not guessed.</p>
          </div>
        </Link>

        {/* Center navigation links */}
        <nav className="hidden md:flex items-center space-x-1 p-1 bg-slate-100/80 dark:bg-slate-800/80 rounded-xl border border-slate-200/60 dark:border-slate-700/60">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = location.pathname.startsWith(link.to);
            const cls = `flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
              isActive
                ? 'bg-white dark:bg-slate-900 text-sky-600 dark:text-sky-400 shadow-sm'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
            }`;
            // The storefront is a separate static site: a normal link loads it, the router would not.
            return link.external ? (
              <a key={link.to} href={link.external} className={cls}>
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </a>
            ) : (
              <Link key={link.to} to={link.to} className={cls}>
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Right side items */}
        <div className="flex items-center space-x-3">
          <div className="hidden sm:flex items-center space-x-1.5 text-xs text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/50 px-2.5 py-1 rounded-full border border-emerald-200/50 dark:border-emerald-800/50">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="font-medium">Privacy Guaranteed</span>
          </div>

          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="p-2 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors border border-slate-200 dark:border-slate-700"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-sky-600" />}
          </button>
        </div>
      </div>
    </header>
  );
};
