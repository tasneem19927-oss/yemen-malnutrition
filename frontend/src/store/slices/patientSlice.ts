import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { patientService } from '../../services/patientService';
import { Patient, Measurement } from '../../types';

interface PatientState {
  patients: Patient[];
  currentPatient: Patient | null;
  measurements: Measurement[];
  isLoading: boolean;
  error: string | null;
}

const initialState: PatientState = {
  patients: [],
  currentPatient: null,
  measurements: [],
  isLoading: false,
  error: null,
};

export const fetchPatients = createAsyncThunk(
  'patients/fetchAll',
  async (_, { rejectWithValue }) => {
    try {
      return await patientService.getAll();
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch patients');
    }
  }
);

export const fetchPatientById = createAsyncThunk(
  'patients/fetchById',
  async (id: number, { rejectWithValue }) => {
    try {
      return await patientService.getById(id);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch patient');
    }
  }
);

const patientSlice = createSlice({
  name: 'patients',
  initialState,
  reducers: {
    clearCurrentPatient: (state) => {
      state.currentPatient = null;
      state.measurements = [];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPatients.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchPatients.fulfilled, (state, action) => {
        state.isLoading = false;
        state.patients = action.payload;
      })
      .addCase(fetchPatients.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchPatientById.fulfilled, (state, action) => {
        state.currentPatient = action.payload;
        state.measurements = action.payload.measurements || [];
      });
  },
});

export const { clearCurrentPatient } = patientSlice.actions;
export default patientSlice.reducer;
