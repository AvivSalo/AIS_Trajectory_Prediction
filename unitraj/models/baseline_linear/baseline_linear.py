import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel


class BaselineLinear(BaseModel):
    """
    Ordinary least-squares (OLS) trajectory extrapolation baseline.

    For each sample, fits a linear position-vs-time model to the valid
    past timesteps of the ego vessel and extrapolates it forward.  There
    are no trainable parameters; the model is purely analytical and serves
    as a classical lower-bound when comparing against Wayformer-AIS.

    All coordinates are kept in the same normalised space (divided by
    position_scale) as the rest of the pipeline.  BaseModel.log_info
    handles de-normalisation before computing metric values.
    """

    def __init__(self, config):
        super().__init__(config)
        self.past_len = config.get('past_len', 300)
        self.future_len = config.get('future_len', 60)
        self.num_modes = config.get('num_modes', 6)

        # Dummy trainable parameter so PyTorch Lightning can build an
        # optimizer without errors.  It never receives a meaningful gradient.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------

    def _linear_extrapolate(
        self, positions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fit an OLS line through valid past positions and extrapolate.

        positions : (past_len, 2)  – normalised x, y in ego-relative frame
        mask      : (past_len,)    – float or bool; non-zero = valid timestep
        returns   : (future_len, 2)
        """
        device = positions.device
        valid = mask.bool()
        n = int(valid.sum().item())

        if n == 0:
            return torch.zeros(self.future_len, 2, device=device)

        if n == 1:
            # Single observation – constant-position prediction
            last_pos = positions[valid][-1]
            return last_pos.unsqueeze(0).expand(self.future_len, -1)

        t = torch.where(valid)[0].float()   # (n,)  absolute timestep indices
        xy = positions[valid]               # (n, 2)

        # OLS: fit xy = a * t + b  independently for x and y
        t_mean = t.mean()
        t_c = t - t_mean                                            # centred
        t_var = (t_c * t_c).sum().clamp(min=1e-8)

        a = (t_c.unsqueeze(1) * xy).sum(dim=0) / t_var             # (2,)
        b = xy.mean(dim=0) - a * t_mean                            # (2,)

        # Extrapolate to future timestep indices
        future_t = torch.arange(
            self.past_len,
            self.past_len + self.future_len,
            dtype=torch.float32,
            device=device,
        )                                                            # (future_len,)
        pred = a.unsqueeze(0) * future_t.unsqueeze(1) + b          # (future_len, 2)
        return pred

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(self, batch):
        inputs = batch['input_dict']
        obj_trajs = inputs['obj_trajs']            # (B, max_agents, past_len, features)
        obj_trajs_mask = inputs['obj_trajs_mask']  # (B, max_agents, past_len)
        track_idx = inputs['track_index_to_predict']  # (B,)

        B = obj_trajs.shape[0]
        device = obj_trajs.device

        pred_list = []
        for b in range(B):
            idx = int(track_idx[b].item())
            pos = obj_trajs[b, idx, :, :2]            # (past_len, 2)
            mask = obj_trajs_mask[b, idx, :]           # (past_len,)
            pred_list.append(self._linear_extrapolate(pos, mask))

        pred_xy = torch.stack(pred_list, dim=0)        # (B, future_len, 2)

        # Build the GMM-format output tensor expected by BaseModel:
        #   channels: x, y, log_std_x, log_std_y, rho
        # Fixed log-std = 0 (std = 1 in normalised space) and rho = 0.
        log_std = torch.zeros(B, self.future_len, 2, device=device)
        rho = torch.zeros(B, self.future_len, 1, device=device)
        pred_full = torch.cat([pred_xy, log_std, rho], dim=-1)  # (B, future_len, 5)

        # Replicate as num_modes identical modes (single-modal baseline)
        predicted_trajectory = pred_full.unsqueeze(1).expand(
            -1, self.num_modes, -1, -1
        ).contiguous()                              # (B, num_modes, future_len, 5)

        # Put all probability mass on mode 0 (all modes are identical anyway)
        predicted_probability = torch.zeros(B, self.num_modes, device=device)
        predicted_probability[:, 0] = 1.0

        prediction = {
            'predicted_probability': predicted_probability,
            'predicted_trajectory': predicted_trajectory,
        }

        # Compute MSE loss (detached – no gradient through the OLS solution)
        gt_trajs = inputs['center_gt_trajs']        # (B, future_len, 2)
        gt_mask = inputs['center_gt_trajs_mask']    # (B, future_len)

        valid = gt_mask.bool()
        if valid.any():
            loss = F.mse_loss(
                pred_xy[valid].detach(),
                gt_trajs[valid].detach(),
            ) + self._dummy.sum() * 0.0
        else:
            loss = self._dummy.sum() * 0.0

        return prediction, loss

    def configure_optimizers(self):
        # Optimizer exists only to satisfy PyTorch Lightning; the dummy
        # parameter never receives a meaningful gradient.
        return optim.AdamW(self.parameters(), lr=self.config.get('learning_rate', 1e-4))