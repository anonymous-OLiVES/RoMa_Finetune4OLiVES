"""Refer to pull request: https://github.com/Parskatt/RoMa/pull/146"""


import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
import tqdm
from romatch.utils.utils import tensor_to_pil

from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--low_path", type=str, required=True)
    parser.add_argument("--normal_path", type=str, required=True)
    parser.add_argument("--save_path", default="aligned_output", type=str)

    args, _ = parser.parse_known_args()

    low_videos_paths = sorted(os.listdir(args.low_path))
    normal_videos_paths = sorted(os.listdir(args.normal_path))

    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(2160, 3840))

    for (low_video, normal_video) in zip(low_videos_paths, normal_videos_paths):
        low_frames = sorted(os.listdir(os.path.join(args.low_path, low_video)))
        normal_frames = sorted(os.listdir(os.path.join(args.normal_path, normal_video)))
        assert len(low_frames) == len(normal_frames), f"Number of frames do not match for {low_video} and {normal_video}"
        for (low_frame, normal_frame) in tqdm.tqdm(zip(low_frames, normal_frames), leave=False, total=len(low_frames)):
            im1_path = os.path.join(args.low_path, low_video, low_frame)
            im2_path = os.path.join(args.normal_path, normal_video, normal_frame)

            # Create model
            """
            DO NOT PASS IN CROPPED VERSION, CROPPING LATER SHOULD REMOVE MOST OF THE EDGE ARTIFACTS
            """

            H, W = roma_model.get_output_resolution()

            im1 = Image.open(im1_path).resize((W, H))
            im2 = Image.open(im2_path).resize((W, H))

            # Match
            warp, certainty = roma_model.match(im1_path, im2_path, device=device)
            # Sampling not needed, but can be done with model.sample(warp, certainty)
            x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
            x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

            im2_transfer_rgb = F.grid_sample(x2[None], warp[:, :, :W, 2:], mode="bilinear", align_corners=False)[0]
            im1_transfer_rgb = F.grid_sample(x1[None], warp[:, :, W:, :2], mode="bilinear", align_corners=False)[0]
            os.makedirs(os.path.join(args.save_path, low_video), exist_ok=True)
            out_path = os.path.join(args.save_path, low_video, low_frame)
            tensor_to_pil(im2_transfer_rgb, unnormalize=False).save(out_path)