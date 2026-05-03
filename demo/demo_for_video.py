import os
import gc  # 添加垃圾回收模块

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil, numpy_to_pil, tensor_to_pil_margin
from romatch.utils.flow_viz import viz_flo
from glob import glob
from romatch import roma_outdoor
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--im_low_dir", default="../assets/00001.png", type=str)
parser.add_argument("--im_norm_dir", default="../assets/00020.png", type=str)
parser.add_argument("--save_dir", default="../demo/", type=str)
parser.add_argument('--W_crop_margin', type=int, default=64)
parser.add_argument('--H_crop_margin', type=int, default=72)

args, _ = parser.parse_known_args()


def clear_gpu_cache():
    """清理GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # 清理进程间通信缓存
    elif torch.backends.mps.is_available():
        # MPS设备的内存管理
        torch.mps.empty_cache()


def print_memory_usage():
    """打印内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(
            f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max Allocated: {max_allocated:.2f}GB")
    elif torch.backends.mps.is_available():
        # MPS目前没有直接的内存查询API
        print("MPS device - memory usage not directly available")


def occ_map(flow1, flow2):
    pass


def process_video(low_video_path):
    # 找对应文件
    video_name = os.path.basename(low_video_path)


def get_name_list(csv_file_path) -> list:
    result = []

    with open(csv_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        available_columns = reader.fieldnames
        if not available_columns:
            raise RuntimeError(f"The csv file {csv_file_path} is empty.")
        normal_col = None
        low_col = None

        for col in available_columns:
            col_lower = col.strip().lower()
            if col_lower == 'normal':
                normal_col = col
            elif col_lower == 'low':
                low_col = col
        # 读取数据
        for row in reader:
            data_dict = {
                "normal": row[normal_col].strip() if row[normal_col] else None,
                "low": row[low_col].strip() if row[low_col] else None
            }
            result.append(data_dict)

    return result


@torch.no_grad()  # 添加装饰器，减少内存使用
def process_frame_pair(roma_model, low_path, normal_path, save_video_path, args):
    """处理单帧图像对"""
    try:
        H, W = roma_model.get_output_resolution()

        # 加载图像
        im_normal = Image.open(normal_path).resize((W, H))
        x2 = (torch.tensor(np.array(im_normal)) / 255).to(device).permute(2, 0, 1)

        # 匹配
        warp, certainty = roma_model.match(low_path, normal_path, device=device)

        # 应用变换
        im2_transfer_rgb = F.grid_sample(
            x2[None], warp[:, :, :W, 2:],
            mode="bilinear", align_corners=False
        )[0]

        # 保存结果图像
        save_path = os.path.join(save_video_path, os.path.basename(normal_path))
        tensor_to_pil_margin(im2_transfer_rgb, False, args.H_crop_margin, args.W_crop_margin).save(save_path)

        # 保存warp和certainty
        warp_np = warp.cpu().numpy()
        certainty_np = certainty.cpu().numpy()
        base_filename = os.path.splitext(os.path.basename(normal_path))[0]

        warp_filename = os.path.join(save_video_path, f"{base_filename}_warp.npy")
        certainty_filename = os.path.join(save_video_path, f"{base_filename}_certainty.npy")

        np.save(warp_filename, warp_np)
        np.save(certainty_filename, certainty_np)

        # 清理临时变量
        del im_normal, x2, warp, certainty, im2_transfer_rgb
        del warp_np, certainty_np

        return True

    except Exception as e:
        print(f"Error processing {normal_path}: {str(e)}")
        return False
    finally:
        # 确保清理缓存
        clear_gpu_cache()


def main():
    # 初始清理
    clear_gpu_cache()

    # Create model
    roma_model = roma_outdoor(
        device=device,
        coarse_res=560,
        upsample_res=(1080 + args.H_crop_margin, 1920 + args.W_crop_margin)
    )

    # 将模型设置为评估模式
    roma_model.eval()

    # Input
    name_list = get_name_list(r"/data1/Dataset/Esprit/Offset/name_list_new.csv")
    low_dir = args.im_low_dir
    normal_dir = args.im_norm_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    total_videos = len(name_list)

    for video_idx, video_pair in enumerate(name_list):
        print(f"\nProcessing video pair {video_idx + 1}/{total_videos}: {video_pair['normal']}")

        # 定期清理缓存
        if video_idx > 0 and video_idx % 5 == 0:
            print("Periodic cache clearing...")
            clear_gpu_cache()
            print_memory_usage()

        low_video_path = os.path.join(low_dir, video_pair["low"])
        normal_video_path = os.path.join(normal_dir, video_pair["normal"])
        save_video_path = os.path.join(save_dir, video_pair["normal"])
        os.makedirs(save_video_path, exist_ok=True)

        low_frm_list = sorted(glob(low_video_path + "/*.png"))
        normal_frm_list = sorted(glob(normal_video_path + "/*.png"))

        frames_num = min(len(low_frm_list), len(normal_frm_list))

        print(f"Found {frames_num} frames in {normal_video_path}")

        # 批量处理的统计
        processed_count = 0
        error_count = 0

        for frm_idx in range(frames_num):
            # 每处理一定帧数清理一次缓存
            if frm_idx > 0 and frm_idx % 20 == 0:
                print(f"  Processed {frm_idx}/{frames_num} frames, clearing cache...")
                clear_gpu_cache()

            low_path = low_frm_list[frm_idx]
            normal_path = normal_frm_list[frm_idx]

            try:
                # 处理单帧
                success = process_frame_pair(roma_model, low_path, normal_path, save_video_path, args)

                if success:
                    processed_count += 1
                else:
                    error_count += 1

                # 每10帧打印一次进度
                if (frm_idx + 1) % 10 == 0:
                    print(f"  Progress: {frm_idx + 1}/{frames_num} frames")

            except Exception as e:
                print(f"  Error in frame {frm_idx}: {str(e)}")
                error_count += 1
                clear_gpu_cache()  # 出错时清理缓存
                continue

        print(f"  Video completed: {processed_count} processed, {error_count} errors")

        # 每个视频处理完后清理缓存
        print("  Clearing cache after video...")
        clear_gpu_cache()

    # 最终清理
    print("\nAll videos processed. Final cache clearing...")
    clear_gpu_cache()

    # 如果是CUDA设备，重置内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\nProcessing completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        clear_gpu_cache()
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        clear_gpu_cache()
        raise