import { apiClient } from './apiService';
import { Patient, Measurement } from '../types';

export const patientService = {
  async getAll(params?: { search?: string; governorate?: string }): Promise<Patient[]> {
    const response = await apiClient.get('/patients', { params });
    return response.data;
  },

  async getById(id: number): Promise<Patient> {
    const response = await apiClient.get(`/patients/${id}`);
    return response.data;
  },

  async create(patient: Partial<Patient>): Promise<Patient> {
    const response = await apiClient.post('/patients', patient);
    return response.data;
  },

  async update(id: number, patient: Partial<Patient>): Promise<Patient> {
    const response = await apiClient.put(`/patients/${id}`, patient);
    return response.data;
  },

  async addMeasurement(patientId: number, measurement: Partial<Measurement>): Promise<Measurement> {
    const response = await apiClient.post(`/patients/${patientId}/measurements`, measurement);
    return response.data;
  },

  async getMeasurements(patientId: number): Promise<Measurement[]> {
    const response = await apiClient.get(`/patients/${patientId}/measurements`);
    return response.data;
  },
};
