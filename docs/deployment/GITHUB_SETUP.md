# GitHub Repository Setup Guide

## 1. Create Repository on GitHub

```bash
# Create new repository on GitHub (do not initialize with README)
# Then run these commands locally:

git init
git add .
git commit -m "Initial commit: Yemen Malnutrition Prediction System v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/yemen-malnutrition-platform.git
git push -u origin main
```

## 2. Configure Repository Settings

### Branch Protection Rules (Main)
- Require pull request reviews before merging
- Require status checks to pass (CI tests)
- Require branches to be up to date before merging
- Include administrators

### Secrets (Settings > Secrets and variables > Actions)
Add these secrets:
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `JWT_SECRET_KEY` - Production JWT secret
- `SECRET_KEY` - Production Django secret

### Environments
Create environments:
- `staging` - For develop branch deployments
- `production` - For main branch deployments

## 3. GitHub Pages (Optional)

For documentation:
```bash
git checkout -b gh-pages
git push origin gh-pages
```

Then enable Pages in Settings > Pages > Source: gh-pages branch

## 4. Issue Labels

Create these labels:
- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvement
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `priority:high` - High priority
- `priority:low` - Low priority

## 5. Project Board

Create a project board with columns:
- Backlog
- To Do
- In Progress
- Review
- Done

## 6. Milestones

Create milestones for releases:
- v1.1.0 - Next features
- v1.2.0 - Future enhancements

## 7. Team Setup

Add collaborators:
- Backend developers
- Frontend developers
- ML engineers
- DevOps engineers

Set permissions:
- Admin: Project leads
- Write: Core developers
- Triage: Contributors

## 8. Webhooks (Optional)

Configure webhooks for:
- Slack/Discord notifications
- Deployment triggers
- External CI systems

## 9. Release Process

```bash
# Create a new release
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

GitHub Actions will automatically:
1. Run tests
2. Build Docker images
3. Push to registry
4. Create GitHub release

## 10. Monitoring

After deployment, monitor:
- GitHub Actions runs
- Docker Hub builds
- Application metrics (Prometheus/Grafana)
- Security alerts (Dependabot)
