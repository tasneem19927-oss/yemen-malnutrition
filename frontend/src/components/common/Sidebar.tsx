import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';
import {
  Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText,
  Divider, Typography, Box
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Assessment as AssessmentIcon,
  MenuBook as MenuBookIcon,
  Description as DescriptionIcon,
  AdminPanelSettings as AdminIcon,
  MedicalServices as NurseIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import { RootState } from '../../store';

const menuItems = [
  { path: '/dashboard', label: 'Dashboard', icon: DashboardIcon, roles: ['admin', 'doctor', 'nurse'] },
  { path: '/patients', label: 'Patients', icon: PeopleIcon, roles: ['admin', 'doctor', 'nurse'] },
  { path: '/predictions', label: 'Predictions', icon: AssessmentIcon, roles: ['admin', 'doctor', 'nurse'] },
  { path: '/knowledge', label: 'Knowledge Base', icon: MenuBookIcon, roles: ['admin', 'doctor'] },
  { path: '/reports', label: 'Reports', icon: DescriptionIcon, roles: ['admin', 'doctor'] },
  { path: '/admin', label: 'Admin', icon: AdminIcon, roles: ['admin'] },
  { path: '/nurse', label: 'Nurse Station', icon: NurseIcon, roles: ['admin', 'nurse'] },
  { path: '/settings', label: 'Settings', icon: SettingsIcon, roles: ['admin', 'doctor', 'nurse'] },
];

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const open = useSelector((state: RootState) => state.ui.sidebarOpen);

  const userRole = user?.role?.name || '';

  const filteredItems = menuItems.filter(item => item.roles.includes(userRole));

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: 260,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 260,
          boxSizing: 'border-box',
          bgcolor: 'background.paper',
        },
      }}
    >
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="h6" color="primary" fontWeight="bold">
          YMN-Predict
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Clinical Decision Support
        </Typography>
      </Box>

      <Divider />

      <List>
        {filteredItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;

          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                selected={isActive}
                onClick={() => navigate(item.path)}
                sx={{
                  '&.Mui-selected': {
                    bgcolor: 'primary.light',
                    '&:hover': { bgcolor: 'primary.light' },
                  },
                }}
              >
                <ListItemIcon>
                  <Icon color={isActive ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      <Box sx={{ mt: 'auto', p: 2, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          v1.0.0 | WHO Compliant
        </Typography>
      </Box>
    </Drawer>
  );
};
