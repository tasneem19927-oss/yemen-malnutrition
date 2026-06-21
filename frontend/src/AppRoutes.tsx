import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';

import { useAuth } from './contexts/AuthContext';
import { Layout } from './components/common/Layout';
import { ProtectedRoute } from './components/common/ProtectedRoute';

// Lazy load pages for code splitting
const LoginPage = lazy(() => import('./pages/LoginPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const PatientsPage = lazy(() => import('./pages/PatientsPage'));
const PatientDetailPage = lazy(() => import('./pages/PatientDetailPage'));
const PredictionsPage = lazy(() => import('./pages/PredictionsPage'));
const PredictionResultPage = lazy(() => import('./pages/PredictionResultPage'));
const KnowledgeBasePage = lazy(() => import('./pages/KnowledgeBasePage'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const AdminPage = lazy(() => import('./pages/AdminPage'));
const NursePage = lazy(() => import('./pages/NursePage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

const LoadingFallback: React.FC = () => (
  <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
    <CircularProgress />
  </Box>
);

const AppRoutes: React.FC = () => {
  const { user } = useAuth();

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={!user ? <LoginPage /> : <Navigate to="/dashboard" />} />

        {/* Protected routes */}
        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/patients" element={<PatientsPage />} />
            <Route path="/patients/:id" element={<PatientDetailPage />} />
            <Route path="/predictions" element={<PredictionsPage />} />
            <Route path="/predictions/:id" element={<PredictionResultPage />} />
            <Route path="/knowledge" element={<KnowledgeBasePage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/settings" element={<SettingsPage />} />

            {/* Role-specific routes */}
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/nurse" element={<NursePage />} />
          </Route>
        </Route>

        {/* Default redirect */}
        <Route path="/" element={<Navigate to={user ? "/dashboard" : "/login"} />} />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Suspense>
  );
};

export default AppRoutes;
