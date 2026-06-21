# An AI model for child Malnutrition prediction in Yemen.

## Overview
AI-powered child malnutrition prediction platform using XGBoost. Predicts stunting, wasting, and underweight in children aged 0-59 months.To increase accuracy, a RAG system was added. To enable it to run on resource-constrained mobile devices, we used BioMobileBERT.
## Key Features

- **XGBoost Prediction Engine**: 3 independent models for Stunting, Wasting, and Underweight
- **BioMobileBERT NER**: Medical entity extraction from clinical texts (Arabic/English)
- **Clinical RAG**: Evidence-based recommendations using WHO/UNICEF guidelines
- **Offline-First**: PWA with service workers, IndexedDB, and local model caching
- **Edge AI**: ONNX export with INT8 quantization for Android, Raspberry Pi, and low-end devices
- **RBAC**: Role-based access control (Admin, Doctor, Nurse)
- **Bilingual**: Full Arabic and English support

## Architecture

```
yemen-malnutrition-platform/
├── backend/          # FastAPI + PostgreSQL + Redis
├── frontend/         # React + TypeScript + Material UI
├── mlops/            # Training and MLOps pipelines
├── edge/             # ONNX edge deployment
├── docker/           # Docker configurations
├── devops/           # CI/CD and monitoring
└── knowledge_base/   # WHO standards and clinical guidelines
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

### Docker Deployment

```bash
# Clone repository
git clone <repository-url>
cd yemen-malnutrition-platform

# Start all services
docker-compose up -d

# Access services
# Backend API: http://localhost:8000
# Frontend: http://localhost:3000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001
```

### Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Training Models

### XGBoost Models
```bash
cd mlops/training/xgboost
python train_xgboost_models.py --data /path/to/yemen_mics6.csv --output ./data/models --optimize
```

### BioMobileBERT NER
```bash
cd mlops/training/biomobilebert
python train_biomobilebert_ner.py --data-dir ./data/annotated --epochs 5 --export-onnx
```

## ONNX Export
```bash
cd edge/onnx
python onnx_export.py --model-dir ./data/models --output-dir ./edge/onnx
```

## Testing

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

## API Documentation

- OpenAPI/Swagger: http://localhost:8000/api/docs (development)
- ReDoc: http://localhost:8000/api/redoc (development)

## Monitoring

- Prometheus metrics at `/metrics`
- Grafana dashboards for system health
- Application logs in `./logs/`

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- WHO Child Growth Standards
- UNICEF Nutrition Strategy
- Yemen MICS6 Dataset
- BioMobileBERT (nlpie/bio-mobilebert)
