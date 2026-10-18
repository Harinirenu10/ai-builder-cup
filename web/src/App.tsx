import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './lib/auth';
import { ApiError } from './lib/api';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { CustomerApp } from './pages/CustomerApp';
import { BusinessConsole } from './pages/BusinessConsole';
import { MockStore } from './pages/MockStore';
import { ConnectPage } from './pages/ConnectPage';
import { ReturnsDesk } from './pages/biz/ReturnsDesk';
import { ChartStudio } from './pages/biz/ChartStudio';
import { SimulatePage } from './pages/biz/SimulatePage';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 30, // 30 seconds
      retry: (failureCount, error) => {
        if (error instanceof ApiError && (error.status === 401 || error.status === 403 || error.status === 404)) {
          return false;
        }
        return failureCount < 2;
      },
      refetchOnWindowFocus: false,
    },
  },
});

const Layout: React.FC = () => (
  <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
    <Navbar />
    <main className="flex-1">
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<CustomerApp />} />
        <Route path="/biz" element={<BusinessConsole />} />
        <Route path="/biz/returns" element={<ReturnsDesk />} />
        <Route path="/biz/charts" element={<ChartStudio />} />
        <Route path="/biz/simulate" element={<SimulatePage />} />
        {/* In production /store/ is the static storefront (web/store); this route is the dev-server fallback. */}
        <Route path="/store" element={<MockStore />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </main>
    <footer className="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-xs text-slate-500">
      <p>Measured Fit &bull; Google Cloud AI Builder Cup &bull; Retail &amp; Commerce</p>
    </footer>
  </div>
);

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* The consent popup opened by the store widget has no app chrome. */}
            <Route path="/app/connect" element={<ConnectPage />} />
            <Route path="*" element={<Layout />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
};

export default App;

