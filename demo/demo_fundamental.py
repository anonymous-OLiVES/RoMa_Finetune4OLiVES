import os.path

from PIL import Image
import torch
import cv2
from romatch import roma_outdoor
import numpy as np
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

def warp_image_with_homography(
        img,
        H: np.ndarray,
        output_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    使用单应性矩阵变换图像
    """
    if output_size is None:
        output_size = (img.shape[1], img.shape[0])

    warped = cv2.warpPerspective(img, H, output_size)
    return warped

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="../assets/00020.png", type=str)
    parser.add_argument("--im_B_path", default="../assets/00001.png", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = roma_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    # F, mask = cv2.findFundamentalMat(
    #     kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=3
    # )

    pass
    H, mask = cv2.findHomography(
        kpts2.cpu().numpy(),
        kpts1.cpu().numpy(),
        method=cv2.RANSAC,
        ransacReprojThreshold=0.7,
        maxIters=3
    )
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    warped_img2 = warp_image_with_homography(img2, H)
    overlap = cv2.addWeighted(img1, 0.5, warped_img2, 0.5, 0)
    cv2.imwrite(os.path.join(os.path.dirname(im1_path), "warped_img.png"), warped_img2)
    cv2.imwrite(os.path.join(os.path.dirname(im1_path), "overlap.png"), overlap)