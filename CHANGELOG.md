# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-06-21

### Added
- Initial release of Yemen Child Malnutrition Prediction System
- FastAPI backend with RBAC (Admin, Doctor, Nurse)
- React + TypeScript frontend with Material UI
- 3 XGBoost models for Stunting, Wasting, Underweight prediction
- BioMobileBERT NER for medical entity extraction (Arabic/English)
- Clinical RAG system with FAISS and WHO guidelines
- PDF report generation (bilingual)
- Offline-first PWA with service workers
- Docker Compose deployment
- Prometheus + Grafana monitoring
- GitHub Actions CI/CD pipeline
- ONNX edge deployment support

### Features
- WHO Child Growth Standards z-score calculation
- Severity classification (Normal/Mild/Moderate/Severe)
- Evidence-based clinical recommendations
- Real-time prediction with confidence scores
- Bilingual support (Arabic/English)
- Role-based dashboard views
- Patient management with measurements
- Knowledge base with approval workflow

### Technical
- PostgreSQL database with SQLAlchemy ORM
- Redis caching
- JWT authentication
- Rate limiting
- Audit logging
- API documentation (OpenAPI/Swagger)
