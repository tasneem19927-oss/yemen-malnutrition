import React, { createContext, useContext, useState, useEffect } from 'react';

interface OfflineContextType {
  isOffline: boolean;
  pendingSync: number;
  queueAction: (action: string, data: any) => void;
  syncNow: () => Promise<void>;
}

const OfflineContext = createContext<OfflineContextType | undefined>(undefined);

export const OfflineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [pendingSync, setPendingSync] = useState(0);

  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const queueAction = (action: string, data: any) => {
    // Store in IndexedDB for later sync
    const queue = JSON.parse(localStorage.getItem('syncQueue') || '[]');
    queue.push({ action, data, timestamp: Date.now() });
    localStorage.setItem('syncQueue', JSON.stringify(queue));
    setPendingSync(queue.length);
  };

  const syncNow = async () => {
    const queue = JSON.parse(localStorage.getItem('syncQueue') || '[]');
    if (queue.length === 0) return;

    // Process queue
    for (const item of queue) {
      try {
        // Send to API
        console.log('Syncing:', item);
      } catch (error) {
        console.error('Sync failed:', error);
      }
    }

    localStorage.removeItem('syncQueue');
    setPendingSync(0);
  };

  return (
    <OfflineContext.Provider value={{ isOffline, pendingSync, queueAction, syncNow }}>
      {children}
    </OfflineContext.Provider>
  );
};

export const useOffline = () => {
  const context = useContext(OfflineContext);
  if (!context) throw new Error('useOffline must be used within OfflineProvider');
  return context;
};
