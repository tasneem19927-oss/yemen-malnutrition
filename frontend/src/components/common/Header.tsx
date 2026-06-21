import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Badge, Box, Avatar, Menu, MenuItem } from '@mui/material';
import { Menu as MenuIcon, Notifications as NotificationsIcon, Language as LanguageIcon } from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useOffline } from '../../contexts/OfflineContext';
import { toggleSidebar, setLanguage } from '../../store/slices/uiSlice';
import { RootState } from '../../store';

export const Header: React.FC = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { isOffline, pendingSync } = useOffline();
  const language = useSelector((state: RootState) => state.ui.language);
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleLanguageToggle = () => {
    dispatch(setLanguage(language === 'en' ? 'ar' : 'en'));
  };

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <IconButton
          color="inherit"
          edge="start"
          onClick={() => dispatch(toggleSidebar())}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Yemen Malnutrition Prediction
        </Typography>

        {isOffline && (
          <Badge color="warning" badgeContent={pendingSync} sx={{ mr: 2 }}>
            <Typography variant="caption" color="warning.light">
              Offline
            </Typography>
          </Badge>
        )}

        <IconButton color="inherit" onClick={handleLanguageToggle}>
          <LanguageIcon />
        </IconButton>

        <IconButton color="inherit">
          <Badge badgeContent={0} color="error">
            <NotificationsIcon />
          </Badge>
        </IconButton>

        <Box sx={{ ml: 2 }}>
          <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
            <Avatar sx={{ bgcolor: 'secondary.main' }}>
              {user?.full_name?.charAt(0) || 'U'}
            </Avatar>
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={() => setAnchorEl(null)}
          >
            <MenuItem onClick={() => navigate('/settings')}>Settings</MenuItem>
            <MenuItem onClick={handleLogout}>Logout</MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  );
};
