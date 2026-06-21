import { configureStore } from '@reduxjs/toolkit';
import authReducer from './store/slices/authSlice';
import patientReducer from './store/slices/patientSlice';
import predictionReducer from './store/slices/predictionSlice';
import uiReducer from './store/slices/uiSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    patients: patientReducer,
    predictions: predictionReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['auth/setUser'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
