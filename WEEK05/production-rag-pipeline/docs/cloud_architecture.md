# Nimbus Cloud — Platform Architecture Reference

**Document ID:** NC-ARCH-0417-ZQ
**Owner:** Platform Infrastructure Guild (team handle `@plat-strato`)
**Classification:** Internal — Engineering Only
**Revision:** 3.8 (supersedes NC-ARCH-0398-ZQ)

---

## 1. Overview

This document describes the reference architecture of the Nimbus Cloud platform as deployed across the **Kestrel-West** and **Kestrel-East** regions. It covers the VPC layout, Kubernetes worker node standards, networking, storage tiers, and the canonical names of internal services. All engineers deploying workloads to Nimbus infrastructure should treat this document as the source of truth; where it conflicts with a team wiki page, this document wins.

The platform is organized around three principles:

1. **Environment isolation** — Staging and Production never share a VPC, a cluster, or a storage bucket.
2. **Declarative everything** — every resource is defined in the `nimbus-infra-zq` repository and applied via pipeline. Console changes are reverted nightly by the drift reaper (`driftwatch-7k`).
3. **Blast-radius limits** — each cluster is capped at 240 worker nodes; larger workloads are sharded across clusters rather than grown vertically.

---

## 2. VPC Layout

### 2.1 VPC Inventory

| VPC Name | Region | CIDR Block | Purpose |
|---|---|---|---|
| Nimbus VPC-Stage | Kestrel-West | `10.84.0.0/16` | Pre-production testing, canary rehearsal |
| Nimbus VPC-Prod | Kestrel-West | `10.112.0.0/16` | Primary production |
| Nimbus VPC-Prod-DR | Kestrel-East | `10.144.0.0/16` | Disaster recovery, warm standby |
| Nimbus VPC-Tooling | Kestrel-West | `10.201.0.0/18` | CI runners, artifact registry, bastions |

### 2.2 Subnet Design

Each workload VPC is divided into three tiers across three availability zones (`kw-az1`, `kw-az2`, `kw-az3`):

| Tier | Subnet Size | Example (VPC-Prod, kw-az1) | Internet Access |
|---|---|---|---|
| Edge | `/22` | `10.112.0.0/22` | Inbound via Lumen Gateway only |
| App | `/19` | `10.112.32.0/19` | Egress via NAT (`nat-qorvo-01..03`) |
| Data | `/21` | `10.112.128.0/21` | None |

The **Data** tier has no route to the internet. Database and cache nodes reach external package mirrors only through the internal mirror `pkgmirror.zq-internal.nimbus`.

### 2.3 Peering and Transit

- VPC-Stage and VPC-Prod are **not peered**. Promotion of artifacts happens through the registry in VPC-Tooling.
- VPC-Prod and VPC-Prod-DR are connected via the **Halyard Transit Hub** (`tgw-halyard-9x2`) with encrypted links at 25 Gbps.
- VPC-Tooling peers with all other VPCs using read-only routes to the registry subnet `10.201.8.0/24`.

```text
            ┌──────────────────────┐
            │   Nimbus VPC-Tooling │
            │   10.201.0.0/18      │
            └─────────┬────────────┘
          peering     │      peering
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
┌───────────┐  ┌────────────┐   ┌──────────────┐
│ VPC-Stage │  │  VPC-Prod  │◄─►│ VPC-Prod-DR  │
│10.84/16   │  │ 10.112/16  │   │ 10.144/16    │
└───────────┘  └────────────┘   └──────────────┘
                  tgw-halyard-9x2 (25 Gbps)
```

---

## 3. Kubernetes Clusters

### 3.1 Cluster Inventory

| Cluster | VPC | Control Plane Version | Max Nodes |
|---|---|---|---|
| `kc-stage-wren` | VPC-Stage | 1.31.4-nim7 | 80 |
| `kc-prod-osprey-a` | VPC-Prod | 1.31.4-nim7 | 240 |
| `kc-prod-osprey-b` | VPC-Prod | 1.31.4-nim7 | 240 |
| `kc-dr-merlin` | VPC-Prod-DR | 1.31.4-nim7 | 160 |

The control plane is managed by the **Corvid** control service, which runs three etcd replicas per cluster on dedicated `cp-ctl-x4` instances outside the worker pool.

### 3.2 Node Pools

| Pool Name | Instance Class | Use Case | Taints |
|---|---|---|---|
| `np-general-v3` | `nmb.g4.xlarge` | Stateless APIs, web | none |
| `np-memory-v2` | `nmb.m4.2xlarge` | Caches, JVM services | `workload=memory:NoSchedule` |
| `np-batch-spot` | `nmb.c4.xlarge` (spot) | Async jobs, ETL | `capacity=spot:NoSchedule` |
| `np-gpu-tern` | `nmb.p2.4xlarge` | Model inference | `accel=gpu:NoSchedule` |

### 3.3 System Requirements

All Kubernetes worker nodes must meet the following minimums. Nodes that fail the bootstrap check (`nodecheck-qv`) are cordoned automatically and do not join the cluster.

#### 3.3.1 Compute and Memory

| Pool | vCPU (min) | RAM (min) | CPU Architecture | Max Pods per Node |
|---|---|---|---|---|
| `np-general-v3` | 8 | 32 GiB | x86_64 (AVX2 required) | 58 |
| `np-memory-v2` | 8 | 64 GiB | x86_64 | 58 |
| `np-batch-spot` | 4 | 16 GiB | x86_64 or arm64 | 29 |
| `np-gpu-tern` | 16 | 128 GiB | x86_64 + 2× accelerator | 29 |

Reserved resources per node (not schedulable by workloads):

- **System reserved:** 0.5 vCPU, 1.5 GiB RAM
- **Kubelet reserved:** 0.25 vCPU, 1 GiB RAM
- **Eviction threshold:** memory available < 750 MiB or nodefs available < 12%

#### 3.3.2 Disk

| Volume | Size | Type | Notes |
|---|---|---|---|
| Root (`/`) | 80 GiB | `vol-ssd-gp` | OS image `nimos-2611-hardened` |
| Container runtime (`/var/lib/containerd`) | 200 GiB | `vol-ssd-gp` | Separate volume, required |
| Ephemeral scratch (`/mnt/scratch`) | 400 GiB | Local NVMe | `np-batch-spot` and `np-gpu-tern` only |

Minimum sustained IOPS for the container runtime volume is **6,000**, with throughput of at least **250 MiB/s**.

#### 3.3.3 Software Baseline

```yaml
# nodecheck-qv baseline profile: worker-baseline-r12
os_image: nimos-2611-hardened
kernel_min: "6.6.41-nim"
containerd: "1.7.22"
kubelet: "1.31.4-nim7"
cni_plugin: "strata-cni 4.2.1"
agents:
  - name: nimbeacon        # metrics and logs shipper
    version: ">=3.14.0"
  - name: sentrylace       # runtime security sensor
    version: ">=2.9.3"
time_sync: chrony (source: ntp.zq-internal.nimbus)
```

---

## 4. Networking

### 4.1 Ingress

All public traffic enters through the **Lumen Gateway** tier — a fleet of L7 proxies (`lumen-gw-01` through `lumen-gw-12`) in the Edge subnets. Lumen terminates TLS using certificates issued by the internal CA **Pallas-CA-R3** for internal names and by the public issuance pipeline for customer-facing names.

- Default idle timeout: 60 seconds
- Max request body: 32 MiB (configurable per route up to 512 MiB)
- WAF rule set: `lumen-waf-ruleset-2611b`

### 4.2 Service Mesh

East–west traffic between services uses the **Trellis** mesh. Every pod receives a Trellis sidecar that enforces mutual TLS. Certificates rotate every 24 hours.

| Setting | Value |
|---|---|
| mTLS mode | STRICT (all namespaces except `kube-system`) |
| Retry budget | 20% of active requests |
| Default timeout | 15 seconds |
| Circuit breaker | 5 consecutive 5xx → eject for 30 s |

### 4.3 DNS

- Internal zone: `zq-internal.nimbus`
- Service records follow `<service>.<namespace>.<cluster>.zq-internal.nimbus`
- Example: `ledgerline.billing.kc-prod-osprey-a.zq-internal.nimbus`

### 4.4 Network Policy

Every namespace starts with a **default-deny** policy. Teams must add explicit allow rules in their service manifest under `trellis.allow`. Cross-namespace calls require approval from the owning team in the `netpol-requests-zq` queue.

---

## 5. Storage

### 5.1 Block Storage

| Class | StorageClass Name | IOPS (baseline) | Use Case |
|---|---|---|---|
| General SSD | `nim-ssd-gp` | 3,000 | Default for PVCs |
| Provisioned SSD | `nim-ssd-iox` | up to 64,000 | Databases |
| Throughput HDD | `nim-hdd-st` | 500 | Log archives, cold scratch |

Snapshots for `nim-ssd-iox` volumes are taken every 4 hours and retained for 14 days.

### 5.2 Object Storage

Object storage is provided by **Basalt**, with buckets named by convention `bsl-<env>-<team>-<purpose>`:

- `bsl-prod-billing-invoices`
- `bsl-stage-platform-artifacts`
- `bsl-prod-obs-logs-archive`

Lifecycle defaults: objects move to cold tier after 30 days and expire after 400 days unless the bucket carries the tag `retention=legal-hold-q7`.

### 5.3 Managed Databases

| Engine | Service Name | Topology |
|---|---|---|
| Relational | **Quarrydb** | 1 primary + 2 sync replicas across AZs |
| Key-value cache | **Flintcache** | 6-shard cluster, 1 replica per shard |
| Event stream | **Rillstream** | 9 brokers, replication factor 3 |

---

## 6. Core Service Catalog

| Service | Namespace | Description | Owning Team |
|---|---|---|---|
| `nimctl-api` | `platform` | Backend for the `nimctl` CLI | Platform Infra |
| `ledgerline` | `billing` | Usage metering and invoice generation | Billing Eng |
| `gatekeep-auth` | `identity` | Token issuance and session validation | Identity |
| `cinder-scheduler` | `platform` | Batch job orchestration | Platform Infra |
| `nimbeacon-collector` | `observability` | Metrics and log aggregation | Observability |
| `quillnotify` | `comms` | Email and webhook notifications | Comms |
| `harrow-registry` | `tooling` | Container image and artifact registry | Dev Productivity |
| `driftwatch-7k` | `platform` | Configuration drift detection and reversal | Platform Infra |

---

## 7. Observability

- **Metrics:** scraped every 15 seconds by `nimbeacon-collector`, retained 13 months.
- **Logs:** shipped to `bsl-prod-obs-logs-archive`, searchable for 30 days in the hot index.
- **Traces:** sampled at 5% by default, 100% for requests carrying header `x-nim-trace-force: 1`.
- **Dashboards:** live in the **Vantage** console under folder `Platform / Core`.

---

## 8. Change Control

Any change to VPC CIDRs, peering, transit, or node pool instance classes requires an **Architecture Change Record (ACR)** filed in the `acr-zq` tracker and approval from two Platform Infra leads. See `deployment_runbook.md` for application-level rollout procedures.
