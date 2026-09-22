// ============================================================
// SERVER API SERVICE LAYER
// Currently returns mock data.
// To connect to the real Python Flower server later:
//   - Replace mock imports with fetch() / axios calls
//   - Replace WebSocket placeholders with real ws:// connections
// ============================================================

import {
  mockKPIData,
  mockTrainingControl,
  mockPerformanceHistory,
  mockPerformanceSummary,
  mockHospitals,
  mockPerformanceMetrics,
  mockUpdateStatistics,
  mockDropoutMonitor,
  mockSystemEvents,
  mockSystemStatus,
} from "../data/mockServerData";

// ----- REST API PLACEHOLDERS -----
// const BASE_URL = import.meta.env.VITE_SERVER_URL || "http://localhost:8080";

export const serverApi = {
  // GET /api/status
  getSystemStatus: async () => mockSystemStatus,

  // GET /api/kpi
  getKPIData: async () => mockKPIData,

  // GET /api/training/control
  getTrainingControl: async () => mockTrainingControl,

  // GET /api/model/performance
  getPerformanceHistory: async () => mockPerformanceHistory,

  // GET /api/model/summary
  getPerformanceSummary: async () => mockPerformanceSummary,

  // GET /api/hospitals
  getHospitals: async () => mockHospitals,

  // GET /api/model/metrics
  getPerformanceMetrics: async () => mockPerformanceMetrics,

  // GET /api/model/updates
  getUpdateStatistics: async () => mockUpdateStatistics,

  // GET /api/dropout
  getDropoutMonitor: async () => mockDropoutMonitor,

  // GET /api/events
  getSystemEvents: async () => mockSystemEvents,

  // POST /api/training/start
  startTraining: async () => ({ success: true }),

  // POST /api/training/pause
  pauseTraining: async () => ({ success: true }),

  // POST /api/training/stop
  stopTraining: async () => ({ success: true }),

  // POST /api/training/reset
  resetTraining: async () => ({ success: true }),
};

// ----- WEBSOCKET PLACEHOLDER -----
// export const connectEventStream = (onMessage) => {
//   const ws = new WebSocket(`ws://${BASE_URL}/ws/events`);
//   ws.onmessage = (e) => onMessage(JSON.parse(e.data));
//   return ws;
// };
