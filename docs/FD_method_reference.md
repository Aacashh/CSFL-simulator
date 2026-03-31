# Federated Distillation (FD) Method Reference

Implementation reference for the FedTSKD and FedTSKD-G algorithms.

**Paper**: Mu, Garg, Ratnarajah. "Federated Distillation in Massive MIMO Networks: Dynamic Training Convergence Analysis and Communication Channel-Aware Learning." *IEEE Trans. Cognitive Commun. Networking*, vol. 10, no. 4, Aug 2024.

---

## 1. Overview

Unlike Federated Learning (FL), which exchanges **model weights** between clients and server, Federated Distillation (FD) exchanges **logits** (soft predictions) on a shared public dataset. Key advantages:

- **Communication efficiency**: ~1% of FL overhead (logits are much smaller than model weights)
- **Model heterogeneity**: each client can maintain a different model architecture
- **Privacy preservation**: no model parameters are shared

## 2. System Model

- **N** users with private datasets D_n, a base station (BS) with N_BS antennas
- Each device has N_D antennas (typically 1)
- Shared public dataset X of size D_p (typically 2000 samples)
- Non-IID data distribution across clients (Dirichlet partitioning)

## 3. Mathematical Formulation

### 3.1 Training Loss (Eq. 1-2)

The overall loss for client n on private data:
```
L_n(w) = (1/D_n) * sum_{(x,y) in D_n} l(F_n(x; w), y)
```
where F_n maps inputs to logits and l is cross-entropy loss.

### 3.2 Distillation Loss (Eq. 3)

KL divergence on the public dataset:
```
Q_n(w; Z) = (1/|Z|) * sum_{(x_p, z_p) in X x Z} KL(LSM(F_n(x_p; w)), SM(z_p))
```
where LSM = log-softmax, SM = softmax.

In implementation with temperature T:
```python
loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(target_logits / T, dim=-1),
    reduction='batchmean'
) * (T * T)
```

### 3.3 Local Training Update (Eq. 12)
```
w_{n,t+1} = w_{n,t} - eta_t * grad L_n(w_{n,t}, D_{n,t}),  t in T_LT
```

### 3.4 Local Distillation Update (Eq. 14)
```
w_{n,t+1} = w_{n,t} - eta_t * grad Q_n(w_{n,t}, Z_hat_{n,t}),  t in T_LD
```

### 3.5 Logit Aggregation (Eq. 17)
```
z_bar_t = sum_{n=1}^{N} lambda_n * z_{n,t} + omega_D
```
where lambda_n = D_n / sum(D_j) are data-size weights.

### 3.6 Dynamic Training Steps (Section V-A)
```
K_r = ceil(D_n / batch_size) * max(1, 5 - floor((r-1) / 25))
```
Near convergence, fewer local iterations are needed, reducing communication.

## 4. Communication Channel Model (mMIMO)

### 4.1 Uplink (Eq. 15-16)

Logits are resized into complex-valued streams and transmitted via mMIMO. With ZF processing at the BS:
```
Z_hat_t = Z_t + omega_{t,UL}
omega_{t,UL} ~ N(0, sigma_UL^2 * I)
sigma_UL^2 = N_D / (2 * SNR_UL * N_BS)
```

### 4.2 Downlink (Eq. 20-22)

Server broadcasts via ZF precoder:
```
omega_{n,DL} ~ N(0, sigma_DL^2 * I)
sigma_DL^2 = var(z_hat) / (2 * SNR_DL)
```

### 4.3 Effective Noise (Eq. 23d, Appendix A)
```
phi_n(z_{n,t}) = z_bar_t + omega_bar_{t,av} + omega_bar_{n,t,DL}
sigma_omega^2 = sigma_UL^2 / N + sigma_DL^2
```

## 5. FedTSKD-G: Group-Based FD (Algorithm 2)

Handles heterogeneous channel conditions by:

1. Splitting clients into "good" and "bad" channel groups (based on channel quality threshold)
2. Aggregating logits per group separately
3. Concatenating: z_hat^[u] = [z_hat^[u,bad], z_hat^[u,good]]
4. Server model's last FC layer doubled to 2*C output neurons
5. After distillation, server sends group-specific logits back to clients

## 6. Convergence (Theorem 1)

Under Assumptions 1-4 (smoothness, convexity, bounded variance, bounded gradients):
```
E{L(w_{n,t+1})} - L_n* <= (psi/2) * nu_n / (t + 1 + beta_1)
```
Converges as O(t^{-1}). Depends on:
- psi: smoothness parameter
- mu: strong convexity parameter
- Statistical properties C_n, C_qn of training and distillation losses

## 7. Implementation in CSFL Simulator

### Key Files
- `core/fd_simulator.py` — FDSimulator class (main FD training loop)
- `core/fd_aggregation.py` — logit_avg(), logit_avg_grouped()
- `core/channel.py` — MIMOChannel (uplink/downlink noise)
- `core/models.py` — FDCNN1, FDCNN2, FDCNN3 (heterogeneous architectures from Table III)

### SimConfig FD Fields
| Field | Default | Paper Value | Description |
|-------|---------|-------------|-------------|
| paradigm | "fl" | "fd" | Training paradigm |
| public_dataset | "same" | "STL-10" | Public dataset for logit exchange |
| public_dataset_size | 2000 | 2000 | Number of public samples |
| distillation_epochs | 2 | 2 | S distillation steps |
| distillation_batch_size | 500 | 500 | Distillation batch size |
| temperature | 1.0 | 1.0 | KL divergence temperature |
| distillation_lr | 0.001 | 0.001 | Adam learning rate |
| dynamic_steps | True | True | Enable K_r decay |
| dynamic_steps_base | 5 | 5 | Initial multiplier |
| dynamic_steps_period | 25 | 25 | Rounds per decrease |
| n_bs_antennas | 64 | 64 | BS antennas |
| ul_snr_db | -8.0 | -8.0 | Uplink SNR (dB) |
| dl_snr_db | -20.0 | -20.0 | Downlink SNR (dB) |
| quantization_bits | 8 | 8 | Logit quantization bits |

### Client Selection Integration
All 47+ existing selection methods work with FD unchanged. The `select_clients()` interface is identical:
```python
def select_clients(round_idx, K, clients, history, rng, time_budget=None, device=None, **kwargs):
    return (selected_ids, scores, state_update)
```

ClientInfo fields (data_size, last_loss, grad_norm, channel_quality, participation_count, label_histogram, etc.) are populated identically in both FL and FD modes.

## 8. Paper Simulation Setup (for reproduction)

| Parameter | Value |
|-----------|-------|
| Users (N) | 15 |
| Rounds (R) | 200 |
| BS Antennas (N_BS) | 64 |
| Device Antennas (N_D) | 1 |
| UL SNR | -8 dB |
| DL SNR | -20 dB |
| Quantization | 8-bit, QPSK |
| Optimizer | Adam, eta=0.001 |
| Batch size (train) | 128 |
| Batch size (distill) | 500 |
| Epochs | 2 (both train and distill) |
| Non-IID | Dirichlet alpha=0.5 |
| Public dataset size | 2000 |
| Model heterogeneity | 3 CNN architectures (1.2M, 79K, 25K params) |
| Datasets | CIFAR-10+STL-10, MNIST+FMNIST |

## 9. References

```bibtex
@article{mu2024fd_mimo,
  author    = {Yuchen Mu and Navneet Garg and Tharmalingam Ratnarajah},
  title     = {Federated Distillation in Massive {MIMO} Networks:
               Dynamic Training Convergence Analysis and Communication
               Channel-Aware Learning},
  journal   = {IEEE Trans. Cognitive Commun. Networking},
  volume    = {10},
  number    = {4},
  pages     = {1535--1550},
  year      = {2024},
  doi       = {10.1109/TCCN.2024.3378215}
}
```
