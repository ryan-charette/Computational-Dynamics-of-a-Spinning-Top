# Security Policy

## Supported versions

Until the first stable release, only the latest commit on `main` is supported.

## Reporting

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting feature when available, or contact the repository
owner privately through the contact information on their GitHub profile.
Include reproduction steps, affected versions, and potential impact.

The project processes local YAML, Parquet, NumPy, and model files. Only load
model artifacts from trusted sources because Python model serialization is not
a safe interchange format.

