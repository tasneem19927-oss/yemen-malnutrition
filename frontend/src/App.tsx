import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { Provider } from 'react-redux';
import { I18nextProvider } from 'react-i18next';
import { ToastContainer } from 'react-toastify';

import { store } from './store';
import { theme } from './styles/theme';
import { i18n } from './i18n';
import AppRoutes from './AppRoutes';
import { AuthProvider } from './contexts/AuthContext';
import { OfflineProvider } from './contexts/OfflineContext';

import 'react-toastify/dist/ReactToastify.css';

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <I18nextProvider i18n={i18n}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AuthProvider>
            <OfflineProvider>
              <BrowserRouter>
                <AppRoutes />
                <ToastContainer position="bottom-right" />
              </BrowserRouter>
            </OfflineProvider>
          </AuthProvider>
        </ThemeProvider>
      </I18nextProvider>
    </Provider>
  );
};

export default App;
