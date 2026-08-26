# Enterprise Cloud Security & IAM Governance Guidelines

## 1. Identity & Access Management (IAM)
- Principle of Least Privilege (PoLP) is strictly enforced across all AWS and GCP cloud accounts.
- Multi-Factor Authentication (MFA) using hardware FIDO2 security keys or Okta Verify is mandatory for all employee accounts.
- Direct root/admin account logins are strictly prohibited and trigger automated alerts to the Security Operations Center (SOC).
- Temporary credentials via AWS IAM Roles / STS with maximum session durations of 1 hour are required for all CLI/SDK operations. Long-lived API keys are disallowed.

## 2. Data Classification and Encryption Standards
- Data is classified into four tiers: Public, Internal, Confidential, and Restricted (PII / PCI-DSS / HIPAA).
- Encryption at Rest: AES-256 (or AWS KMS with customer-managed keys CMK) is required for all S3 buckets, RDS databases, DynamoDB tables, and EBS volumes.
- Encryption in Transit: TLS 1.3 is enforced across all internal and public network endpoints. TLS 1.0 and 1.1 are permanently deprecated.
- Database backups must be encrypted and retained according to the 7-year regulatory compliance schedule in geo-redundant storage.

## 3. Vulnerability Management & Incident Response
- Container images in AWS ECR must pass automated Trivy / Snyk security scans before deployment to Kubernetes clusters.
- Critical severity vulnerabilities (CVE score >= 9.0) must be patched within 24 hours of notification.
- High severity vulnerabilities (CVE 7.0 - 8.9) must be remediated within 7 business days.
- Security incidents must be reported immediately via Slack channel `#security-incidents` or by paging the On-Call Incident Commander via PagerDuty.
