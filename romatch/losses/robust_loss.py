from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from romatch.utils.utils import get_gt_warp
import wandb
import romatch
import math
import torch.distributed as dist
import os

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        # self.cossim = nn.CosineSimilarity(dim=1)

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def info_nce_loss(self, x, y, temperature=0.07):
        # normalize
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        N = x.size(0)

        # similarity matrix
        logits = torch.mm(x, y.t()) / temperature

        labels = torch.arange(N, device=x.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.t(), labels)

        return (loss_i + loss_j) / 2

    def forward(self, corresps, batch):
        scales_total = list(corresps.keys())
        scales = [s for s in scales_total if isinstance(s, int)]
        scales_cl = [s for s in scales_total if isinstance(s, str)]
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()

        if len(scales_cl) > 0:
            for s in scales_cl:
                cf = corresps[s]
                x = cf["z1"]
                y = cf["z2"]
                num_features = x.shape[-1]
                batch_size = x.shape[0]
                sim_coeff = 0.25
                std_coeff = 0.25
                cov_coeff = 0.01
                info_coeff = 10

                repr_loss = F.mse_loss(x, y)

                if int(os.environ.get('WORLD_SIZE', 1)) > 1:
                    x = torch.cat(FullGatherLayer.apply(x), dim=0)
                    y = torch.cat(FullGatherLayer.apply(y), dim=0)
                    batch_size *= dist.get_world_size()
                    # print(f"Batch size: {batch_size}")

                info_nce = self.info_nce_loss(x, y)
                # print(f"info_nce: {info_nce}")

                x = x - x.mean(dim=0)
                y = y - y.mean(dim=0)

                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                std_y = torch.sqrt(y.var(dim=0) + 0.0001)
                std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

                cov_x = (x.T @ x) / (batch_size - 1)
                cov_x = off_diagonal(cov_x).pow_(2).float()
                cov_y = (y.T @ y) / (batch_size - 1)
                cov_y = off_diagonal(cov_y).pow_(2).float()
                cov_loss = (cov_x.sum().div(num_features)
                            + cov_y.sum().div(num_features))
                cov_loss = torch.clamp(cov_loss, max=100)

                cl = (
                        sim_coeff * repr_loss
                        + std_coeff * std_loss
                        + cov_coeff * cov_loss
                )
                cl += info_nce
                # print(f"cl: {cl}")

                tot_loss = tot_loss + cl / len(scales_cl)
            # print(f"tot_loss: {tot_loss}")
        return tot_loss
