# Enterprise Software Engineering & Architecture Standards

## 1. Code Review & Pull Request Standards
- Every Pull Request (PR) requires at least two peer approvals before merging into `main` or `release` branches.
- PRs must have a descriptive title adhering to Conventional Commits format (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
- Code coverage for new business logic must remain at or above 80% with unit tests.
- Static analysis checks (flake8, ruff, black, mypy for Python; ESLint and Prettier for TypeScript) must pass in GitHub Actions CI without warnings.

## 2. API Design & Versioning
- All microservices must expose RESTful APIs following OpenAPI 3.0 specification guidelines.
- Endpoints must use URL versioning in the path (e.g., `/api/v1/resources`).
- Standard HTTP status codes must be used: 200 OK for successful retrieval, 201 Created for resource generation, 400 Bad Request for validation errors, 401 Unauthorized, 403 Forbidden, 404 Not Found, and 500 Internal Server Error.
- All JSON response envelopes must include standard fields: `data`, `status`, `message`, and timestamp.

## 3. Microservice Architecture & Resilience
- Services communicate asynchronously using Apache Kafka for event-driven workflows and gRPC/REST for synchronous internal RPC.
- Retries on external HTTP dependencies must use exponential backoff with jitter and a maximum of 3 attempts.
- Circuit breaker patterns (e.g., via Resilience4j or Polly) must be configured for all downstream third-party SaaS integrations.
- Rate limiting is enforced at the Envoy / Kong API Gateway level at 1,000 requests/minute per authenticated client token.

## 4. Observability, Logging, and Distributed Tracing
- Structured JSON logging is mandatory. Logs must include `trace_id`, `span_id`, `service_name`, and `environment`.
- Sensitive data (PII, credentials, access tokens, customer payment details) MUST be masked before logging using the enterprise Sanitizer utility.
- Distributed tracing is powered by OpenTelemetry exporting to Datadog / Jaeger.
