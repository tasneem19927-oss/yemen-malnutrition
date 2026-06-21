import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
  sidebarOpen: boolean;
  language: 'en' | 'ar';
  theme: 'light' | 'dark';
  offlineMode: boolean;
  notifications: Array<{ id: string; message: string; type: 'success' | 'error' | 'warning' | 'info' }>;
}

const initialState: UIState = {
  sidebarOpen: true,
  language: (localStorage.getItem('language') as 'en' | 'ar') || 'en',
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'light',
  offlineMode: false,
  notifications: [],
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setLanguage: (state, action: PayloadAction<'en' | 'ar'>) => {
      state.language = action.payload;
      localStorage.setItem('language', action.payload);
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
      localStorage.setItem('theme', action.payload);
    },
    setOfflineMode: (state, action: PayloadAction<boolean>) => {
      state.offlineMode = action.payload;
    },
    addNotification: (state, action: PayloadAction<{ message: string; type: 'success' | 'error' | 'warning' | 'info' }>) => {
      state.notifications.push({
        id: Date.now().toString(),
        ...action.payload,
      });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
  },
});

export const { toggleSidebar, setLanguage, setTheme, setOfflineMode, addNotification, removeNotification } = uiSlice.actions;
export default uiSlice.reducer;
