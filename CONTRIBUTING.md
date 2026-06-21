# Contributing to Yemen Malnutrition Prediction System

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/yemen-malnutrition-platform.git`
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Code Style

- Python: Follow PEP 8
- TypeScript: Follow ESLint rules
- Commit messages: Use conventional commits

## Testing

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Code of Conduct

Be respectful and constructive in all interactions.
