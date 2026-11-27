## Threshold Calibration (12 in-domain, 12 out-of-domain)

Top cross-encoder score = sigmoid(top logit) after Stage 1 (K1=20) and Stage 2 re-ranking. Raw queries, no rewriting.

### Score separation

- In-domain scores: min 0.2300, median 0.9756, max 0.9999
- Out-of-domain scores: min 0.0000, median 0.0014, max 0.4185
- Separation: overlap: highest out-of-domain 0.4185 ≥ lowest in-domain 0.2300

### Error rates by threshold

| Threshold | False fallback (in → web, of 12) | Missed fallback (out → internal, of 12) | Routing accuracy |
|---|---|---|---|
| 0.10 | 0 (0%) | 2 (17%) | 92% |
| 0.20 | 0 (0%) | 1 (8%) | 96% |
| 0.30 | 1 (8%) | 1 (8%) | 92% |
| 0.40 | 1 (8%) | 1 (8%) | 92% |
| 0.50 | 1 (8%) | 0 (0%) | 96% |
| 0.55 **← chosen** | 1 (8%) | 0 (0%) | 96% |
| 0.60 | 1 (8%) | 0 (0%) | 96% |
| 0.70 | 2 (17%) | 0 (0%) | 92% |
| 0.80 | 2 (17%) | 0 (0%) | 92% |
| 0.90 | 3 (25%) | 0 (0%) | 88% |

### Per-query scores (routed at 0.55)

| Query | Label | Top score | Route @ 0.55 | Correct? | Top chunk |
|---|---|---|---|---|---|
| How long do customers have to dispute an invoice? | in | 0.9999 | INTERNAL | ✅ | `billing_policy.md` |
| What are the system requirements for Kubernetes worker nodes? | in | 0.9998 | INTERNAL | ✅ | `cloud_architecture.md` |
| What is the overage rate per vCPU-hour on the Tempest plan? | in | 0.9988 | INTERNAL | ✅ | `billing_policy.md` |
| What is the CIDR block of Nimbus VPC-Stage? | in | 0.9921 | INTERNAL | ✅ | `cloud_architecture.md` |
| What are the canary rollout stages and bake times? | in | 0.9896 | INTERNAL | ✅ | `deployment_runbook.md` |
| Which service handles usage metering and invoice generation? | in | 0.9756 | INTERNAL | ✅ | `cloud_architecture.md` |
| How do I roll back a release after a full rollout? | in | 0.9694 | INTERNAL | ✅ | `deployment_runbook.md` |
| What does NIM-E4021 mean? | in | 0.9499 | INTERNAL | ✅ | `deployment_runbook.md` |
| How much does the Gale plan cost per month? | in | 0.9454 | INTERNAL | ✅ | `billing_policy.md` |
| How do I fix a stale Terraform state lock? | in | 0.8988 | INTERNAL | ✅ | `deployment_runbook.md` |
| What happens when an account is 15 days past due? | in | 0.6110 | INTERNAL | ✅ | `billing_policy.md` |
| Can I get my money back if I cancel a yearly prepaid plan after a month? | in | 0.2300 | WEB | ❌ | `billing_policy.md` |
| What is the default idle timeout of an AWS Application Load Balancer? | out | 0.4185 | WEB | ✅ | `cloud_architecture.md` |
| What is the latest stable version of Kubernetes? | out | 0.1951 | WEB | ✅ | `cloud_architecture.md` |
| How do I write a Terraform module for an S3 bucket? | out | 0.0674 | WEB | ✅ | `deployment_runbook.md` |
| What is Google Cloud's uptime SLA for Compute Engine? | out | 0.0229 | WEB | ✅ | `billing_policy.md` |
| What is the maximum execution timeout for AWS Lambda? | out | 0.0053 | WEB | ✅ | `cloud_architecture.md` |
| How much does an AWS t3.medium instance cost per hour? | out | 0.0014 | WEB | ✅ | `billing_policy.md` |
| How do I install Docker on Ubuntu? | out | 0.0000 | WEB | ✅ | `deployment_runbook.md` |
| How do I cache dependencies in a GitHub Actions workflow? | out | 0.0000 | WEB | ✅ | `cloud_architecture.md` |
| What is the difference between TCP and UDP? | out | 0.0000 | WEB | ✅ | `cloud_architecture.md` |
| What caused the AWS us-east-1 outage in October 2025? | out | 0.0000 | WEB | ✅ | `deployment_runbook.md` |
| What is the weather in Seattle today? | out | 0.0000 | WEB | ✅ | `cloud_architecture.md` |
| Who founded HashiCorp? | out | 0.0000 | WEB | ✅ | `cloud_architecture.md` |
