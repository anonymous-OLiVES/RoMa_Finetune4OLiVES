import os

import cv2

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil
from romatch.utils.flow_viz import viz_flo

from romatch import roma_outdoor
from occ import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="../assets/low_1021.png", type=str)
    parser.add_argument("--im_B_path", default="../assets/normal_970.png", type=str)
    parser.add_argument("--save_path", default="../demo/", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(1152, 1984))

    H, W = roma_model.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    # x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    # x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    # im2_transfer_rgb = F.grid_sample(x2[None], warp[:, :, :W, 2:][None], mode="bilinear", align_corners=False)[0]
    # im1_transfer_rgb = F.grid_sample(x1[None], warp[:, :, W:, :2][None], mode="bilinear", align_corners=False)[0]

    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    flow1 = warp[:, :, :W, 2:]
    flow2 = warp[:, :, W:, :2]
    flow_viz1 = viz_flo(flow1)
    flow_viz2 = viz_flo(flow2)
    cv2.imwrite(f'{save_path}/flow_viz1.png', flow_viz1)
    cv2.imwrite(f'{save_path}/flow_viz2.png', flow_viz2)


    im2_transfer_rgb = F.grid_sample(x2[None], warp[:, :, :W, 2:], mode="bilinear", align_corners=False)[0]
    im1_transfer_rgb = F.grid_sample(x1[None], warp[:, :, W:, :2], mode="bilinear", align_corners=False)[0]

    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)

    white_im = torch.ones((H,2*W),device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im

    save_path1 = os.path.join(save_path, os.path.basename(im1_path))
    save_path2 = os.path.join(save_path, os.path.basename(im2_path))
    tensor_to_pil(im1_transfer_rgb, unnormalize=False).save(save_path1)
    tensor_to_pil(im2_transfer_rgb, unnormalize=False).save(save_path2)
    pass

    hs, ws = flow1.shape[1:3]
    im_A_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws),
        ),
        indexing="ij",
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]), dim=-1).numpy()

    flow_B_A = torch.asarray(warp[:, :, W:, :2][0].cpu().numpy() - im_A_coords).unsqueeze(0).permute(0,3,1,2) / 2
    flow_A_B = torch.asarray(warp[:, :, :W, 2:][0].cpu().numpy() - im_A_coords).unsqueeze(0).permute(0,3,1,2) / 2

    flow_B_A[:,0:1,:,:] *= 1984
    flow_B_A[:,1:2,:,:] *= 1152
    flow_A_B[:,0:1,:,:] *= 1984
    flow_A_B[:,1:2,:,:] *= 1152

    occ_model = Occ_Check_Model()
    occ_f, occ_b = occ_model(flow_B_A, flow_A_B)
    occ_f = occ_f[0, 0].cpu().numpy()
    occ_b = occ_b[0, 0].cpu().numpy()

    occ_map, dist = compute_occlusion_map(flow_B_A, flow_A_B, threshold=1.0)

    pass
