# Security Policy

We take the security of evalseed seriously. If you believe you have found a security vulnerability, please report it responsibly.

## Supported Versions

Only the latest minor release of evalseed receives security updates.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, please report them via GitHub's private vulnerability reporting:

- Go to the Security tab of this repository.
- Click "Report a vulnerability".
- Fill out the form with as much detail as possible.

You should receive an initial response within 7 days. If the issue is confirmed, we will work on a fix and coordinate disclosure with you.

## What to Include

To help us triage faster, please include:

- A clear description of the vulnerability and its impact
- Steps to reproduce, including any required configuration or sample inputs
- The affected version(s) of evalseed
- Any proof-of-concept code or screenshots, if applicable
- Your suggested mitigation, if you have one

## Scope

In scope:

- The evalseed Python package and its source code in this repository
- Official examples and documentation that ship in this repo
- CI workflows defined under .github/workflows

Out of scope:

- Third-party LLM providers, vector stores, or RAG frameworks integrated with evalseed
- Issues that require an attacker to already have local code execution on the user's machine
- Vulnerabilities in dependencies (please report those upstream; we will track via Dependabot)

## Safe Harbor

We will not pursue or support legal action against researchers who:

- Make a good-faith effort to follow this policy
- Avoid privacy violations, data destruction, or service disruption
- Give us reasonable time to investigate and remediate before any public disclosure

Thank you for helping keep evalseed and its users safe.
