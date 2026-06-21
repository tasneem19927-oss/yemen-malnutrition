import { apiClient } from './apiService';
import { Prediction, PredictionRequest, PredictionResult } from '../types';

export const predictionService = {
  async predict(request: PredictionRequest): Promise<PredictionResult> {
    const response = await apiClient.post('/predictions/predict', request);
    return response.data;
  },

  async getByPatient(patientId: number): Promise<Prediction[]> {
    const response = await apiClient.get(`/predictions/patient/${patientId}`);
    return response.data;
  },

  async reviewPrediction(predictionId: number, review: { doctor_notes: string; approved: boolean }): Promise<void> {
    await apiClient.post(`/predictions/${predictionId}/review`, review);
  },

  async generateReport(predictionId: number, language: 'en' | 'ar' = 'en'): Promise<Blob> {
    const response = await apiClient.post(
      '/predictions/reports/generate',
      { prediction_id: predictionId, language },
      { responseType: 'blob' }
    );
    return response.data;
  },
};
