import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { CustomerApp } from './pages/CustomerApp';
import { BusinessConsole } from './pages/BusinessConsole';
import { MockStore } from './pages/MockStore';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/app" element={<CustomerApp />} />
            <Route path="/biz" element={<BusinessConsole />} />
            <Route path="/store" element={<MockStore />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <footer className="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-xs text-slate-500">
          <p>Measured Fit &bull; Google Cloud AI Builder Cup &bull; Retail &amp; Commerce</p>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
