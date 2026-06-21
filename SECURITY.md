# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@yemen-malnutrition.org

Do NOT open public issues for security vulnerabilities.

We will respond within 48 hours and work on a fix.

## Security Measures

- JWT tokens with refresh mechanism
- Rate limiting on all endpoints
- Input validation with Pydantic
- SQL injection prevention via SQLAlchemy ORM
- XSS protection via React escaping
- HTTPS enforcement in production
