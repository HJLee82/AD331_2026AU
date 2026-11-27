# Nimbus Cloud — Service Deployment Runbook

**Document ID:** NC-RUN-0522-ZQ
**Owner:** Release Engineering (team handle `@rel-cirrus`)
**Classification:** Internal — Engineering Only
**Applies to:** All services on `kc-stage-wren`, `kc-prod-osprey-a`, `kc-prod-osprey-b`

---

## 1. Purpose and Scope

This runbook covers the standard procedure for deploying a service to Nimbus Cloud, rolling it out through canary stages, rolling back a bad release, and responding to deployment-related incidents. It assumes the service is already onboarded to `harrow-registry` and has a valid manifest (`nimbus.svc.yaml`) in its repository.

Infrastructure changes (VPCs, node pools, transit) are out of scope; see `cloud_architecture.md` §8.

---

## 2. Prerequisites

Before starting any production deploy, confirm the following:

- [ ] You hold the `deployer-prod` role in `gatekeep-auth` (check with `nimctl whoami --roles`).
- [ ] The release artifact has passed the **Sable** CI pipeline with status `GREEN`.
- [ ] The same artifact digest has run in `kc-stage-wren` for at least **2 hours** without alerts.
- [ ] No deploy freeze is active (`nimctl freeze status`).
- [ ] An on-call engineer from the owning team is reachable for the duration of the rollout.

### 2.1 CLI Setup

```bash
# Install or upgrade the CLI (requires access to VPC-Tooling)
curl -sSL https://pkgmirror.zq-internal.nimbus/nimctl/install.sh | bash
nimctl version            # must be >= 5.12.0

# Authenticate
nimctl login --sso --realm nimbus-eng-zq
nimctl context use kc-prod-osprey-a
```

---

## 3. Standard Deployment

### Step 1 — Validate the manifest

```bash
nimctl manifest lint ./nimbus.svc.yaml
nimctl manifest diff ./nimbus.svc.yaml --env prod
```

The diff must show only the intended image digest and config changes. Unexpected changes to resource limits or network policy must be reviewed by the owning team lead before proceeding.

### Step 2 — Create a release record

```bash
nimctl release create \
  --service ledgerline \
  --artifact harrow-registry.zq-internal.nimbus/billing/ledgerline@sha256:7f3c...e19a \
  --ticket REL-88214 \
  --notes "Adds proration for mid-cycle plan upgrades"
```

This returns a release ID in the format `rls-<service>-<6 chars>`, e.g. `rls-ledgerline-q4m8tz`.

### Step 3 — Deploy to staging (rehearsal)

```bash
nimctl deploy rls-ledgerline-q4m8tz --env stage --wait
nimctl health rls-ledgerline-q4m8tz --env stage
```

Health must report `READY` for all replicas and `smoke: pass`.

### Step 4 — Begin production canary

Proceed to §4. **Never** deploy directly to 100% in production except under an approved emergency change (see §6.4).

---

## 4. Canary Rollout

### 4.1 Canary Stages

| Stage | Traffic Share | Minimum Bake Time | Auto-Promote? |
|---|---|---|---|
| C1 | 1% | 10 min | No |
| C2 | 10% | 20 min | Yes, if gates pass |
| C3 | 25% | 30 min | Yes, if gates pass |
| C4 | 50% | 30 min | No — manual approval |
| Full | 100% | — | — |

### 4.2 Starting the Canary

```bash
# Stage C1
nimctl deploy rls-ledgerline-q4m8tz --env prod --canary 1

# After 10 minutes with clean gates, move to C2
nimctl deploy rls-ledgerline-q4m8tz --env prod --canary 10

# Enable automatic promotion through C3
nimctl canary auto-promote rls-ledgerline-q4m8tz --until 25
```

### 4.3 Canary Gates

The **Tollgate** analyzer compares canary pods with baseline pods every 60 seconds. A stage passes only if all gates hold for the full bake time:

| Gate | Threshold |
|---|---|
| HTTP 5xx rate | ≤ baseline + 0.5 percentage points |
| p99 latency | ≤ baseline × 1.20 |
| Pod restarts | 0 |
| CPU saturation | ≤ 80% of limit |
| Custom SLO gate | as defined in `nimbus.svc.yaml` → `tollgate.slo` |

Check gate status at any time:

```bash
nimctl canary status rls-ledgerline-q4m8tz
```

Example output:

```text
RELEASE               STAGE  TRAFFIC  BAKE      GATES
rls-ledgerline-q4m8tz C2     10%      14m/20m   5/5 PASS
```

### 4.4 Manual Promotion to Full

```bash
nimctl canary approve rls-ledgerline-q4m8tz --stage C4
# after 30 min at 50%:
nimctl deploy rls-ledgerline-q4m8tz --env prod --canary 100
nimctl release finalize rls-ledgerline-q4m8tz
```

Finalizing a release retires the previous ReplicaSet after a **6-hour** grace period. Until then, rollback is instant.

---

## 5. Rollback

### 5.1 When to Roll Back

Roll back immediately — do not debug forward — if any of these occur:

- Tollgate reports a gate failure for two consecutive checks.
- Error code **NIM-E4021** (canary health divergence) or **NIM-E5003** (replica crash loop) is raised.
- Customer-facing errors are reported by Support for the affected service.
- You are unsure. Rolling back is cheap; a prolonged outage is not.

### 5.2 Rollback Commands

```bash
# During canary: drain canary traffic to 0 and stop the release
nimctl canary abort rls-ledgerline-q4m8tz --reason "p99 regression"

# After full rollout (within 6-hour grace window)
nimctl rollback ledgerline --env prod --to previous

# Rollback to a specific earlier release
nimctl rollback ledgerline --env prod --to rls-ledgerline-k2p9wd
```

Verify:

```bash
nimctl release list ledgerline --env prod --limit 3
nimctl health ledgerline --env prod
```

### 5.3 Database Migrations

Releases that include schema migrations against **Quarrydb** must follow the expand/contract pattern. Contract (destructive) migrations may only ship in a release *after* the expanding release has been finalized for 7 days. If a rollback is needed after a migration, file the release ID with the Data Platform on-call before rolling back.

---

## 6. Error Code Reference

| Code | Meaning | Typical Action |
|---|---|---|
| **NIM-E1007** | Manifest schema invalid | Run `nimctl manifest lint`; fix fields |
| **NIM-E2310** | Artifact digest not found in `harrow-registry` | Confirm push succeeded; check digest |
| **NIM-E2318** | Artifact not signed by Sable pipeline | Rebuild through Sable; unsigned images are blocked |
| **NIM-E3044** | Deploy freeze active | Wait, or request exemption (§6.4) |
| **NIM-E3109** | Insufficient role for target env | Request `deployer-prod` in `gatekeep-auth` |
| **NIM-E4021** | Canary health divergence | Abort canary (§5.2) |
| **NIM-E4025** | Tollgate analyzer unreachable | Pause rollout; do not promote blind |
| **NIM-E4400** | Node pool capacity exhausted | Contact Platform Infra; check `np-*` quotas |
| **NIM-E5003** | Replica crash loop | Roll back; collect logs with `nimctl logs --previous` |
| **NIM-E5190** | Trellis sidecar injection failed | Verify namespace label `trellis.inject=enabled` |
| **NIM-E6602** | Legacy Terraform state lock held | See §7.3 |

### 6.4 Emergency Changes

Emergency (non-canary) deploys require an **EC ticket** approved by the Incident Commander and use:

```bash
nimctl deploy rls-ledgerline-q4m8tz --env prod --emergency --ec EC-30471
```

All emergency deploys are reviewed in the weekly Release Council.

---

## 7. Legacy Terraform Modules

Some older services still provision supporting resources (buckets, queues, DNS records) through Terraform modules that predate the current `nimbus-infra-zq` pipeline.

### 7.1 Module Inventory

| Module | Status | Replacement | Sunset Date |
|---|---|---|---|
| `tf-nimbus-v1` | Deprecated | `nim-infra-kit/core` | 2027-03-31 |
| `tf-nimbus-v1-dns` | Deprecated | `nim-infra-kit/dns` | 2027-03-31 |
| `tf-nimbus-basalt-legacy` | Frozen (no new usage) | `nim-infra-kit/objstore` | 2027-01-15 |
| `tf-nimbus-v2` | Supported | — | — |

### 7.2 Working with tf-nimbus-v1

`tf-nimbus-v1` pins an old provider and must be run from the dedicated runner image:

```bash
nimctl tf shell --image tf-runner-legacy:0.13.7-zq
cd infra/legacy/ledgerline
terraform init -backend-config=backend-prod.hcl
terraform plan -out=plan.tfout
```

```hcl
module "ledgerline_queue" {
  source        = "git::ssh://harrow-registry.zq-internal.nimbus/infra/tf-nimbus-v1.git//queue?ref=v1.9.4"
  name          = "ledgerline-usage-events"
  retention_hrs = 72
  env           = "prod"
}
```

Do **not** upgrade the provider version in place; migrations to `nim-infra-kit` are tracked in `INFRA-MIG-2027`.

### 7.3 Resolving NIM-E6602 (state lock)

```bash
nimctl tf locks --workspace ledgerline-prod
# Confirm the lock owner is not running an apply, then:
nimctl tf unlock --workspace ledgerline-prod --lock-id <ID> --reason "stale lock after runner eviction"
```

Never force-unlock a lock less than 30 minutes old without contacting its owner.

---

## 8. Incident Checklist

Use this checklist whenever a deploy causes, or is suspected of causing, customer impact.

### 8.1 First 5 Minutes

- [ ] **Declare** the incident: `nimctl incident open --sev 2 --service ledgerline`
- [ ] **Stop the bleeding**: abort canary or roll back (§5.2).
- [ ] **Freeze** further deploys to the service: `nimctl freeze set ledgerline --env prod`
- [ ] **Assign roles**: Incident Commander, Ops Lead, Comms Lead.

### 8.2 Severity Levels

| Sev | Definition | Response Time | Status Page Update |
|---|---|---|---|
| Sev 1 | Full outage or data loss risk | 5 min | Within 15 min |
| Sev 2 | Major degradation, >5% of requests failing | 15 min | Within 30 min |
| Sev 3 | Minor degradation, workaround available | 1 hour | Optional |
| Sev 4 | No customer impact | Next business day | No |

### 8.3 Stabilization

- [ ] Confirm error rate is back to baseline in Vantage (`Platform / Core / Service Health`).
- [ ] Capture evidence: `nimctl logs ledgerline --env prod --since 1h > incident-logs.txt`
- [ ] Record the timeline in the incident channel `#inc-<incident-id>`.
- [ ] Notify Billing Eng if usage metering may have been affected (credits may apply — see `billing_policy.md` §7).

### 8.4 Closure

- [ ] Lift the freeze: `nimctl freeze clear ledgerline --env prod`
- [ ] Close the incident: `nimctl incident resolve <incident-id>`
- [ ] Schedule a blameless review within **5 business days** for Sev 1–2.
- [ ] File follow-up actions with label `postinc-zq`.
