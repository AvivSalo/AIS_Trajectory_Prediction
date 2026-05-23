# AIS-ACNet — Architecture & Adaptation Notes

> **Reference:** Shin Y., Kim N., Lee H., In S.Y., Hansen M., Yoon Y. (2024).  
> *Deep learning framework for vessel trajectory prediction using auxiliary tasks and convolutional networks.*  
> Engineering Applications of Artificial Intelligence, 132, 107936.  
> https://doi.org/10.1016/j.engappai.2024.107936  
>
> **Original code:** https://github.com/yuyolshin/AIS-ACNet

---

## 1. Motivation

Recurrent models such as LSTM suffer from error accumulation over long prediction horizons and vanishing gradient problems. AIS-ACNet replaces the recurrent encoder with **dilated causal convolutions** (inspired by WaveNet), which process the full input sequence in parallel and can capture long-range temporal dependencies via exponentially growing receptive fields. A second novelty is **multi-task learning**: the model is jointly trained to predict future vessel positions *and* future vessel dynamics (Speed Over Ground and Course Over Ground), which provides additional supervisory signal that regularises the learned representations.

---

## 2. Original Architecture

### 2.1 Dual-Encoder Design

AIS-ACNet employs two parallel convolutional encoders built on the WaveNet backbone:

| Encoder | Input features | Role |
|---------|---------------|------|
| **main-net** | Latitude, Longitude | Learns spatial trajectory patterns |
| **aux-net** | Speed Over Ground (SOG), Course Over Ground (COG) | Learns vessel dynamics |

Both encoders share the same layer structure but have independent weights. They process the same sequence of `T` historical timesteps and produce representations that are fused at every layer.

### 2.2 WaveNet Encoder Layers

Each encoder consists of `L` stacked **dilated causal convolution** layers. At layer `l` with dilation factor `d_l`:

```
residual  =  x_{l-1}
filter    =  tanh ( W_f * x_{l-1} )        [dilated causal conv]
gate      =  σ   ( W_g * x_{l-1} )        [dilated causal conv]
x_gated   =  filter ⊙ gate                [gated activation]
skip_l    =  W_skip * x_gated             [1×1 conv → skip channels]
x_l       =  W_res  * x_gated + residual  [1×1 conv + residual shortcut]
```

The **gated activation** (`tanh ⊙ σ`) is the key building block of WaveNet: the tanh controls the signal magnitude while the sigmoid acts as a learned gate, giving the model expressive power equivalent to LSTM cells without sequential computation.

The **dilated causal convolution** with kernel size `k=2` and dilation `d` looks back exactly one step at distance `d` in the past:

```
output[t] = W[0] · x[t]  +  W[1] · x[t - d]
```

Stacking layers with increasing dilation `d ∈ {1, 2, 4, …}` grows the **receptive field** exponentially while keeping the number of parameters linear in `L`.

### 2.3 Feature Fusion

Between each pair of encoder layers, a **gating-based fusion layer** allows aux-net information to influence the main-net representation:

```
x_fuse   = W_1 · x_main           [1×1 conv on main residual output]
x_a_fuse = W_2 · x_aux            [1×1 conv on aux residual output]
z        = σ ( x_fuse + x_a_fuse ) [learned gate]
x_main   = W_out · ( z ⊙ x_fuse + (1−z) ⊙ x_a_fuse ) + residual
```

The gate `z` adaptively controls how much auxiliary (dynamics) information flows into the main (position) encoder at each layer. This is the core mechanism that distinguishes AIS-ACNet from a plain WaveNet: vessel dynamics are not simply appended as extra input channels but are *continuously integrated* throughout the depth of the network.

### 2.4 Skip Connection Aggregation

All `L` skip outputs are accumulated into a single running sum:

```
skip = Σ_{l=1}^{L} skip_l
```

Because each dilated conv reduces the temporal dimension (without padding), the accumulated `skip` tensor collapses to a single time step by the final layer. This single representation encodes the full history of the input sequence.

### 2.5 Output Heads

Two separate two-layer MLP heads are applied to the skip aggregation:

| Head | Input | Output | Predicts |
|------|-------|--------|---------|
| **Main head** | `skip_main` | `(T_p, 2)` | Future latitude & longitude |
| **Aux head** | `skip_aux` | `(T_p, 2)` | Future SOG & COG |

The output dimension `T_p` is embedded directly in the 1×1 convolution weights, so all future steps are decoded in a single non-autoregressive pass.

### 2.6 Multi-Task Loss

The total training objective is:

```
L = L_d  +  α · L_SOG  +  β · L_COG
```

where:
- `L_d` = Haversine distance loss on predicted positions (main task)
- `L_SOG` = Masked MSE on predicted Speed Over Ground (auxiliary task 1)
- `L_COG` = Masked MSE on predicted Course Over Ground (auxiliary task 2)
- `α = β = 0.2` (ablation in the paper shows this gives best performance)

The intuition is that forcing the model to also explain *how fast* and *in what direction* a vessel will move yields more physically grounded intermediate representations, which in turn improve positional accuracy.

### 2.7 Original Hyperparameters

| Hyperparameter | Original value |
|---|---|
| Hidden / residual channels | 32 |
| Skip channels | 256 (= 32 × 8) |
| End (output MLP) channels | 512 (= 32 × 16) |
| Kernel size | 2 |
| Encoder blocks | 5 |
| Layers per block | 2 |
| Total layers `L` | 10 |
| Dilation schedule per block | [1, 2] |
| Receptive field | 16 steps |
| Input length `T` | 15 min (at 1-min intervals) |
| Prediction horizon `T_p` | 15 min |
| Optimizer | Adam, lr = 0.001, wd = 0.0001 |
| Training epochs | 200 |
| Batch size | 32 |

---

## 3. Adaptations for this Project

The original model was designed for **15-step sequences at 1-minute intervals** evaluated on the Port of Houston AIS dataset. This project uses **5-minute prediction horizons at 1 Hz** (300-step sequences) on a Danish maritime AIS dataset. Four targeted adaptations were made to port the model into the UniTraj framework while preserving the core architecture.

### Adaptation 1 — Dilation Schedule for Long Sequences

**Problem:** The original dilation schedule `[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]` (repeated within 5 blocks of 2 layers) gives a receptive field of only **16 timesteps**, sufficient for 15-step inputs but unable to cover 300-step inputs.

**Solution:** Replace the block-based schedule with an **exponential dilation schedule** across the same 10 layers:

```
Original:    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]   →  RF = 16
This project: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  →  RF = 1024
```

The receptive field formula for kernel size `k=2` is:

```
RF = 1 + Σ d_l = 1 + 1023 = 1024
```

This covers the full 300-step history with substantial margin, matching the spirit of the WaveNet design principle (exponential coverage with `O(L)` parameters and `O(L)` compute). The number of layers and total parameter count (~1.2M) remain identical to the original.

### Adaptation 2 — Single Ego-Vessel Processing

**Problem:** The original model is designed as a **scene-level** model: it takes a crowd of vessels simultaneously (shape `(B, 4, N, T)`, where `N` is the number of vessels) and predicts all of them in one forward pass. In this project we adopt the **marginal (ego-centric) prediction** paradigm used by Wayformer and TrAISformer: one forward pass predicts one target vessel.

**Solution:** Fix `N = 1` and extract only the ego vessel's features from the UniTraj batch dict. The Conv2d kernel `(1, k)` operates purely along the time axis; the vessel dimension is a no-op for `N=1`, so the architecture is mathematically unchanged.

```python
# UniTraj batch → ego features → (B, 4, 1, T)
ego_trajs = obj_trajs[batch_idx, track_index_to_predict]   # (B, T, 39)
pos = ego_trajs[:, :, 0:2]   # x, y
vel = ego_trajs[:, :, 2:4]   # vx, vy
inp = cat([pos, vel], dim=-1).permute(0,2,1).unsqueeze(2)  # (B, 4, 1, T)
```

### Adaptation 3 — Coordinate Space and Loss Function

**Problem:** The original model works with absolute geographic coordinates (latitude / longitude in radians) and uses the **Haversine distance** as the main loss, which requires de-normalising predictions back to radians before computing great-circle distances.

**Solution:** This project's pipeline normalises all positions to an **ego-relative Cartesian frame** (origin = ego vessel's last observed position, units = 100 m). This is consistent with Wayformer, TrAISformer, and the linear baseline. The main loss is therefore:

```
L_d = MSE( pred_xy[valid], gt_trajs[valid] )
```

computed entirely in the normalised space. The UniTraj evaluation framework (`BaseModel.log_info`) de-normalises by `position_scale = 100 m` to report ADE/FDE in metres, ensuring a fair comparison across all models.

### Adaptation 4 — Auxiliary Task Targets

**Problem:** The original model's auxiliary tasks predict future SOG and COG, for which the original dataset provides explicit labels. The UniTraj AIS dataset stores ground-truth future data as positions only (`center_gt_trajs`: shape `(B, T_p, 2)`); future speed and heading are not directly available.

**Solution:** Auxiliary targets are derived on-the-fly from **finite differences** of the ground-truth positions:

```
Δx[t] = gt_x[t] − gt_x[t−1]    (normalised position units / timestep)
Δy[t] = gt_y[t] − gt_y[t−1]
```

The aux-net output heads predict `(Δx, Δy)` — i.e. the normalised velocity vector — instead of raw SOG/COG. This is a natural proxy: `Δx = vx · Δt / position_scale` and `Δy = vy · Δt / position_scale`. The auxiliary loss becomes:

```
L_aux = α · MSE(pred_vx, Δx_gt)  +  β · MSE(pred_vy, Δy_gt)
```

The first timestep of each prediction window is excluded from the auxiliary mask because `Δx[0]` is undefined (no previous future step). The weights `α = β = 0.2` are kept identical to the paper's best-performing ablation setting.

---

## 4. Summary of Changes

| Aspect | Original paper | This project |
|--------|---------------|-------------|
| Input length `T` | 15 steps (1 min/step) | 300 steps (1 Hz) |
| Prediction horizon `T_p` | 15 steps | 300 steps |
| Dilation schedule | `[1,2] × 5 blocks` → RF = 16 | `[1,2,4,…,512]` → RF = 1024 |
| Number of layers | 10 | 10 (unchanged) |
| Parameters | ~1.2M | ~1.2M (unchanged) |
| Vessels per forward pass | `N` (scene-level) | 1 (ego-centric) |
| Coordinate frame | Absolute lat/lon (radians) | Ego-relative Cartesian (normalised) |
| Main loss | Haversine distance | MSE in normalised space |
| Aux targets | SOG, COG (explicit labels) | `Δx, Δy` (derived from position diffs) |
| Aux weights α, β | 0.2, 0.2 | 0.2, 0.2 (unchanged) |
| Optimizer | Adam, lr=0.001 | Adam, lr=0.001 (unchanged) |
| Dataset | Port of Houston (USA) | Danish AIS (North Sea) |

---

## 5. Model Comparison Context

In the broader comparison conducted in this thesis, AIS-ACNet represents the **convolutional multi-task learning** paradigm, contrasted with:

| Model | Paradigm | Key mechanism |
|-------|----------|---------------|
| **Linear baseline** | Classical | OLS extrapolation — no learning |
| **TrAISformer** | Autoregressive Transformer | Discretised token generation |
| **Wayformer** | Non-autoregressive Transformer | Cross-attention encoder–decoder, GMM output |
| **AIS-ACNet** | Non-autoregressive CNN | Dilated causal conv + multi-task auxiliary loss |

The main architectural contrasts relevant for the thesis discussion:

- **Parallelism vs. sequentiality:** AIS-ACNet and Wayformer process the full input in one pass; TrAISformer generates step-by-step. Parallel models avoid error accumulation but cannot condition each future step on its predecessor.
- **Inductive biases:** AIS-ACNet's dilated convolutions are translation-equivariant in time and exploit local temporal structure. Wayformer's self-attention is permutation-invariant and must learn temporal order from positional encodings.
- **Multi-task regularisation:** AIS-ACNet is the only model in this comparison that explicitly regularises the encoder by supervising on vessel dynamics (velocity), not just future positions.
- **Prediction mode:** AIS-ACNet is deterministic (single mode). Wayformer produces a multimodal GMM. TrAISformer can produce K samples via temperature-controlled decoding.

---

## 6. Usage

Create a top-level config that references the method config:

```yaml
# e.g. unitraj/configs/config_ais_acnet_5min.yaml
exp_name: 'ais_acnet_5min_v1'
past_len: 300
future_len: 300
normalize_data: True
train_data_path: ["/path/to/processed_ais/train"]
val_data_path:   ["/path/to/processed_ais/val"]

defaults:
  - method: ais_acnet
```

Then train from `unitraj/`:

```bash
python train.py --config configs/config_ais_acnet_5min.yaml
```

Key tunable hyperparameters in `configs/method/ais_acnet.yaml`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `nhid` | 32 | Residual channel width (scales params ~quadratically) |
| `dilation_factors` | `[1,2,4,…,512]` | Receptive field coverage |
| `aux_sog_weight` α | 0.2 | Weight of vx auxiliary loss |
| `aux_cog_weight` β | 0.2 | Weight of vy auxiliary loss |
| `learning_rate` | 0.001 | Adam LR |
| `train_batch_size` | 32 | GPU memory vs. gradient noise trade-off |