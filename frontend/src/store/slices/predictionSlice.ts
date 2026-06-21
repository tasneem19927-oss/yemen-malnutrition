import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { predictionService } from '../../services/predictionService';
import { Prediction, PredictionRequest } from '../../types';

interface PredictionState {
  predictions: Prediction[];
  currentPrediction: Prediction | null;
  isLoading: boolean;
  error: string | null;
}

const initialState: PredictionState = {
  predictions: [],
  currentPrediction: null,
  isLoading: false,
  error: null,
};

export const createPrediction = createAsyncThunk(
  'predictions/create',
  async (request: PredictionRequest, { rejectWithValue }) => {
    try {
      return await predictionService.predict(request);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Prediction failed');
    }
  }
);

export const fetchPatientPredictions = createAsyncThunk(
  'predictions/fetchByPatient',
  async (patientId: number, { rejectWithValue }) => {
    try {
      return await predictionService.getByPatient(patientId);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch predictions');
    }
  }
);

const predictionSlice = createSlice({
  name: 'predictions',
  initialState,
  reducers: {
    clearCurrentPrediction: (state) => {
      state.currentPrediction = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(createPrediction.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(createPrediction.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentPrediction = action.payload;
        state.predictions.push(action.payload);
      })
      .addCase(createPrediction.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchPatientPredictions.fulfilled, (state, action) => {
        state.predictions = action.payload;
      });
  },
});

export const { clearCurrentPrediction } = predictionSlice.actions;
export default predictionSlice.reducer;
