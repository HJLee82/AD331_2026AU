# Nimbus Cloud — Billing and Pricing Policy

**Document ID:** NC-BIL-0309-ZQ
**Owner:** Billing Engineering & Revenue Operations (team handle `@rev-nimbo`)
**Classification:** Internal — Confidential
**Effective Date:** 2026-12-01
**Metering Service:** `ledgerline` (namespace `billing`)

---

## 1. Purpose

This policy defines how Nimbus Cloud prices its services, how usage above plan allowances is charged, when invoices are issued and due, and when customers are eligible for refunds or service credits. Support, Sales, and Billing Engineering must apply these rules consistently. Exceptions require written approval as described in §9.

All prices are in **US dollars (USD)** and exclude applicable taxes.

---

## 2. Pricing Tiers

### 2.1 Plan Summary

| Plan | Code | Monthly Fee | Included vCPU-hours | Included Storage | Included Egress | Support Level |
|---|---|---|---|---|---|---|
| Spark | `PLN-SPK-01` | $0 | 750 | 25 GB | 50 GB | Community only |
| Drift | `PLN-DRF-02` | $49 | 5,000 | 250 GB | 500 GB | Email, 24-hr response |
| Gale | `PLN-GLE-03` | $399 | 50,000 | 2 TB | 5 TB | Email + chat, 4-hr response |
| Tempest | `PLN-TMP-04` | $2,499 | 400,000 | 20 TB | 40 TB | 24/7, 1-hr Sev 1 response |
| Stratos (Enterprise) | `PLN-STR-05` | Custom (min. $15,000) | Contracted | Contracted | Contracted | Named TAM, 15-min Sev 1 response |

### 2.2 Annual Commitment Discounts

Customers who prepay for 12 months receive a discount on the monthly fee only (not on overage):

| Plan | Annual Prepay Price | Effective Discount |
|---|---|---|
| Drift | $499 / year | 15.1% |
| Gale | $3,999 / year | 16.5% |
| Tempest | $24,990 / year | 16.7% |

Stratos contracts are negotiated individually; the minimum term is 24 months.

### 2.3 Plan Limits

| Limit | Spark | Drift | Gale | Tempest |
|---|---|---|---|---|
| Max clusters | 1 | 3 | 10 | 40 |
| Max worker nodes per cluster | 3 | 20 | 80 | 240 |
| Max Basalt buckets | 5 | 50 | 500 | 2,000 |
| SLA uptime commitment | None | 99.5% | 99.9% | 99.95% |

Spark accounts that exceed included usage are **suspended**, not billed overage, until the next cycle or until the customer upgrades.

---

## 3. Usage Metering

`ledgerline` collects usage events from `nimbeacon-collector` every **5 minutes** and aggregates them hourly. The billable unit for each resource is:

| Resource | Billable Unit | Rounding |
|---|---|---|
| Compute | vCPU-hour | Per second, minimum 60 seconds per instance start |
| Memory (above 4 GiB per vCPU) | GiB-hour | Per second |
| Block storage | GB-month | Prorated hourly |
| Object storage (Basalt) | GB-month | Prorated daily |
| Egress | GB | Per GB, rounded up to nearest 0.01 GB |
| Load balancer (Lumen) | LB-hour + per million requests | Hourly |
| GPU (`np-gpu-tern`) | Accelerator-hour | Per minute |

Internal traffic between availability zones within a VPC is free. Traffic between Nimbus VPC-Prod and Nimbus VPC-Prod-DR over the Halyard Transit Hub is billed at the inter-region rate.

---

## 4. Overage Rules

### 4.1 Overage Rates

Usage beyond a plan's included allowance is charged at the following rates:

| Resource | Drift | Gale | Tempest |
|---|---|---|---|
| vCPU-hour | $0.0420 | $0.0380 | $0.0310 |
| Memory GiB-hour (excess) | $0.0055 | $0.0049 | $0.0041 |
| Block storage (GB-month) | $0.095 | $0.088 | $0.072 |
| Basalt object storage (GB-month) | $0.024 | $0.021 | $0.018 |
| Egress (GB) | $0.085 | $0.070 | $0.055 |
| Inter-region transfer (GB) | $0.030 | $0.025 | $0.020 |
| GPU accelerator-hour | $2.85 | $2.60 | $2.25 |
| Lumen LB per million requests | $0.75 | $0.65 | $0.50 |

Stratos overage rates are set in each contract, with a floor of 85% of Tempest rates.

### 4.2 Overage Thresholds and Notifications

`quillnotify` sends automated alerts to the account's billing contacts at these points:

| Trigger | Notification |
|---|---|
| 75% of any included allowance | Informational email |
| 100% of any included allowance | Email + in-console banner |
| Overage charges reach 50% of monthly fee | Email to billing and technical contacts |
| Overage charges reach 200% of monthly fee | Email + account manager outreach (Gale and above) |

### 4.3 Overage Caps

Customers on Drift and above may set a **spend cap** in the console (Billing → Spend Controls). When the cap is reached:

- New resource creation is blocked with error **NIM-B7104**.
- Existing resources continue to run for a **72-hour grace period**.
- After the grace period, compute resources are scaled to zero; storage is retained for 30 days.

The minimum spend cap that can be set is **110% of the plan's monthly fee**.

### 4.4 Burst Allowance

Gale and Tempest accounts receive a one-time **burst allowance** per billing cycle: the first 10% of vCPU-hour overage is billed at 50% of the standard overage rate. Unused burst allowance does not carry over.

---

## 5. Invoice Schedule

### 5.1 Billing Cycle

- Billing cycles run from the **1st day of the month at 00:00 UTC** to the last day of the month at 23:59:59 UTC.
- Accounts created mid-month are prorated for the plan fee based on the number of remaining days.

### 5.2 Invoice Timeline

| Event | Timing |
|---|---|
| Usage period closes | Last day of month, 23:59:59 UTC |
| Draft invoice generated by `ledgerline` | 1st of next month, by 06:00 UTC |
| Usage reconciliation window | 1st–3rd of month |
| Final invoice issued | 4th of month, by 12:00 UTC |
| Payment due (card on file) | Charged automatically on the 4th |
| Payment due (invoice / wire, Tempest & Stratos) | Net 30 from issue date |

Invoice numbers use the format `INV-<YYYYMM>-<account short ID>-<sequence>`, e.g. `INV-202612-AQ7ZK-0001`.

### 5.3 Late Payment

| Days Past Due | Action |
|---|---|
| 1 | Reminder email |
| 7 | Second reminder; late fee of 1.5% of outstanding balance applied |
| 15 | Account placed in **restricted** mode — no new resources (error **NIM-B7230**) |
| 30 | Compute suspended; storage retained read-only |
| 60 | Account referred to collections; data scheduled for deletion after a further 30 days |

Late fees are capped at 1.5% per month and are never applied to disputed amounts while a dispute is open.

### 5.4 Disputes

Customers may dispute any invoice line within **45 days** of the invoice issue date. Billing Engineering must acknowledge a dispute within 2 business days and resolve it within 15 business days. Disputes are tracked in the `bildisp-zq` queue.

---

## 6. Refund Policy

### 6.1 Eligible Refunds

| Scenario | Refund |
|---|---|
| Plan downgrade or cancellation within 7 days of first paid charge | 100% of the plan fee |
| Annual prepay cancelled within 30 days | 100% minus usage consumed at monthly rates |
| Annual prepay cancelled after 30 days | No refund; service continues until term end |
| Duplicate charge caused by billing error | 100%, processed within 5 business days |
| Metering error confirmed by Billing Engineering | 100% of the overcharged amount |

### 6.2 Non-Refundable Items

- Overage charges for usage that actually occurred
- GPU reservations once the reservation window has started
- Stratos contract minimums
- Charges older than 90 days, except confirmed metering errors

### 6.3 Refund Method

Refunds go to the original payment method. Where that is not possible, the refund is issued as account credit. Refunds over **$5,000** require approval from the Revenue Operations manager.

---

## 7. Service Credits (SLA)

When monthly uptime falls below the plan's SLA commitment, the customer is eligible for a service credit against the plan fee for that month:

| Monthly Uptime | Gale Credit | Tempest Credit | Stratos Credit |
|---|---|---|---|
| < SLA but ≥ 99.0% | 10% | 10% | 15% |
| < 99.0% but ≥ 95.0% | 25% | 30% | 35% |
| < 95.0% | 50% | 50% | 100% |

Drift customers receive a flat 10% credit for any month below 99.5%.

**Conditions:**

- Customers must request credits within **30 days** of the end of the affected month.
- Credits are applied to the next invoice and cannot be converted to cash.
- Total credits in a month may not exceed 100% of that month's plan fee.
- Downtime caused by customer configuration, scheduled maintenance announced 72 hours in advance, or suspension for non-payment is excluded.

Billing Engineering is notified automatically of Sev 1 and Sev 2 incidents via the incident checklist in `deployment_runbook.md` §8.3, so that proactive credits can be reviewed.

---

## 8. Taxes and Currency

- Prices exclude sales tax, VAT, and GST; these are calculated by the **Taxwright** module in `ledgerline` based on the billing address.
- Invoices may be displayed in EUR or GBP for convenience, but are settled in USD at the rate fixed on the 1st of the billing month.

---

## 9. Exceptions and Approvals

| Exception Amount | Approver |
|---|---|
| Up to $500 | Support Team Lead |
| $500.01 – $5,000 | Billing Engineering Manager |
| $5,000.01 – $50,000 | Revenue Operations Director |
| Above $50,000 | VP Finance |

All exceptions must be logged in `bilexc-zq` with the account ID, invoice number, amount, and justification.
