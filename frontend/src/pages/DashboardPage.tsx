import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  Grid, Card, CardContent, Typography, Box, LinearProgress,
  Chip, IconButton, Tooltip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  People as PeopleIcon,
  Warning as WarningIcon,
  Assessment as AssessmentIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';
import { Chart as ChartJS, ArcElement, Tooltip as ChartTooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { RootState } from '../store';
import { useAuth } from '../contexts/AuthContext';

ChartJS.register(ArcElement, ChartTooltip, Legend);

const DashboardPage: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { user } = useAuth();

  // Mock data - would come from API in production
  const stats = {
    totalPatients: 1247,
    newPatients: 89,
    totalPredictions: 3421,
    severeCases: 156,
    moderateCases: 423,
    mildCases: 678,
    normalCases: 990,
  };

  const severityData = {
    labels: ['Normal', 'Mild', 'Moderate', 'Severe'],
    datasets: [{
      data: [stats.normalCases, stats.mildCases, stats.moderateCases, stats.severeCases],
      backgroundColor: ['#4CAF50', '#FF9800', '#FF5722', '#D32F2F'],
      borderWidth: 0,
    }],
  };

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
    trend?: string;
    onClick?: () => void;
  }> = ({ title, value, icon, color, trend, onClick }) => (
    <Card sx={{ cursor: onClick ? 'pointer' : 'default', '&:hover': onClick ? { boxShadow: 4 } : {} }} onClick={onClick}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography color="textSecondary" variant="body2" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" fontWeight="bold">
              {value}
            </Typography>
            {trend && (
              <Typography variant="caption" color="success.main">
                +{trend} this month
              </Typography>
            )}
          </Box>
          <Box
            sx={{
              bgcolor: `${color}15`,
              borderRadius: 2,
              p: 1.5,
              color: color,
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        {t('dashboard.title')}
      </Typography>
      <Typography color="textSecondary" gutterBottom>
        Welcome back, {user?.full_name}
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Patients"
            value={stats.totalPatients}
            icon={<PeopleIcon fontSize="large" />}
            color="#0066CC"
            trend={stats.newPatients.toString()}
            onClick={() => navigate('/patients')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Predictions"
            value={stats.totalPredictions}
            icon={<AssessmentIcon fontSize="large" />}
            color="#4CAF50"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Severe Cases"
            value={stats.severeCases}
            icon={<WarningIcon fontSize="large" />}
            color="#D32F2F"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Moderate Cases"
            value={stats.moderateCases}
            icon={<TrendingUpIcon fontSize="large" />}
            color="#FF5722"
          />
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Severity Distribution
              </Typography>
              <Box sx={{ height: 300, display: 'flex', justifyContent: 'center' }}>
                <Doughnut data={severityData} options={{ maintainAspectRatio: false }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Predictions
              </Typography>
              <Box sx={{ mt: 2 }}>
                {[
                  { patient: 'Ahmad Mohammed', risk: 'severe', time: '2 min ago' },
                  { patient: 'Fatima Ali', risk: 'moderate', time: '15 min ago' },
                  { patient: 'Omar Hassan', risk: 'mild', time: '1 hour ago' },
                  { patient: 'Aisha Saleh', risk: 'normal', time: '2 hours ago' },
                ].map((item, index) => (
                  <Box
                    key={index}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      py: 1.5,
                      borderBottom: index < 3 ? '1px solid #eee' : 'none',
                    }}
                  >
                    <Box>
                      <Typography variant="body1">{item.patient}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {item.time}
                      </Typography>
                    </Box>
                    <Chip
                      label={item.risk.toUpperCase()}
                      color={
                        item.risk === 'severe' ? 'error' :
                        item.risk === 'moderate' ? 'warning' :
                        item.risk === 'mild' ? 'default' : 'success'
                      }
                      size="small"
                    />
                  </Box>
                ))}
              </Box>
              <Box sx={{ mt: 2, textAlign: 'right' }}>
                <IconButton onClick={() => navigate('/predictions')}>
                  <ArrowForwardIcon />
                </IconButton>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item>
                  <Chip
                    label="New Patient"
                    color="primary"
                    onClick={() => navigate('/patients')}
                    sx={{ cursor: 'pointer' }}
                  />
                </Grid>
                <Grid item>
                  <Chip
                    label="Run Prediction"
                    color="secondary"
                    onClick={() => navigate('/predictions')}
                    sx={{ cursor: 'pointer' }}
                  />
                </Grid>
                <Grid item>
                  <Chip
                    label="View Reports"
                    onClick={() => navigate('/reports')}
                    sx={{ cursor: 'pointer' }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
