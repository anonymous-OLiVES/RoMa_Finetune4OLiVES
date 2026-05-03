import torch.nn as nn
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import cv2

# from utils.utils import flow_viz, cvt_ts2np, viz_flo, warp_img, forward_backward_consistency_check


DEVICE = 'cpu'



def bilinear_interpolate(image, x, y):
    """
    双线性插值函数
    Args:
        image: 输入图像或光流场，shape (H, W) 或 (H, W, C)
        x, y: 坐标数组，shape相同，为浮点数坐标
    Returns:
        插值后的值，shape与x,y相同（若image为多通道则增加最后一维）
    """
    H, W = image.shape[:2]

    # 获取四个角点的整数坐标
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # 确保坐标在图像范围内
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # 计算权重
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # 处理多通道情况
    if len(image.shape) == 3:
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]

    # 插值计算
    interpolated = (wa * image[y0, x0] +
                    wb * image[y0, x1] +
                    wc * image[y1, x0] +
                    wd * image[y1, x1])

    return interpolated


def compute_occlusion_map(flow_forward, flow_backward, threshold=1.0):
    """
    计算遮挡图（前向后向一致性方法）

    Args:
        flow_forward: 前向光流 F_{t->t+1}, shape (H, W, 2)
        flow_backward: 后向光流 F_{t+1->t}, shape (H, W, 2)
        threshold: 一致性误差阈值，默认1.0像素

    Returns:
        occlusion_map: 遮挡图，1表示遮挡，0表示可见
        fb_error: 前向后向一致性误差图
    """
    flow_forward = flow_forward[0].permute(1, 2, 0).cpu().numpy()
    flow_backward = flow_backward[0].permute(1, 2, 0).cpu().numpy()

    H, W, _ = flow_forward.shape

    # 创建坐标网格
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)

    # 1. 将t帧的每个像素用前向光流投影到t+1帧
    x_f = x_coords + flow_forward[..., 0]
    y_f = y_coords + flow_forward[..., 1]

    # 2. 采样t+1帧的后向光流（需要双线性插值）
    # 注意：flow_backward的坐标是相对于t+1帧的
    u_b = bilinear_interpolate(flow_backward[..., 0], x_f, y_f)
    v_b = bilinear_interpolate(flow_backward[..., 1], x_f, y_f)

    # 3. 映射回t帧坐标
    x_b = x_f + u_b
    y_b = y_f + v_b

    # 4. 计算一致性误差
    dist = np.sqrt((x_b - x_coords) ** 2 + (y_b - y_coords) ** 2)

    # 5. 阈值判断
    occlusion_map = dist > threshold

    return occlusion_map.astype(np.float32), dist


def visualize_results(img_t, flow_forward, occlusion_map):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原图
    axes[0, 0].imshow(img_t)
    axes[0, 0].set_title('Frame t')
    axes[0, 0].axis('off')

    # 显示前向光流（颜色编码）
    def flow_to_color(flow):
        # 将光流转换为HSV颜色表示
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    flow_color = flow_to_color(flow_forward)
    axes[0, 1].imshow(flow_color)
    axes[0, 1].set_title('Forward Flow (Color)')
    axes[0, 1].axis('off')

    # 显示光流分量
    axes[0, 2].imshow(flow_forward[..., 0], cmap='coolwarm')
    axes[0, 2].set_title('Flow X Component')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(flow_forward[..., 1], cmap='coolwarm')
    axes[1, 0].set_title('Flow Y Component')
    axes[1, 0].axis('off')

    # 显示遮挡图
    axes[1, 1].imshow(occlusion_map, cmap='gray')
    axes[1, 1].set_title('Occlusion Map (White=Occluded)')
    axes[1, 1].axis('off')

    # 显示原图叠加遮挡图
    overlay = img_t.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    overlay[occlusion_map > 0.5] = [1, 0, 0]  # 红色标记遮挡区域
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Occlusion Overlay (Red=Occluded)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()



def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = cv2.resize(img, [640,360])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def torch_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # print(grid.shape,flo.shape,'...')
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
    # tools.check_tensor(x, 'x')
    # tools.check_tensor(vgrid, 'vgrid')
    output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
    # mask = torch.autograd.Variable(torch.ones(x.size()))
    # if x.is_cuda:
    #     mask = mask.cuda()
    # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
    #
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1
    # output = output * mask
    # # nchw->>>nhwc
    # if x.is_cuda:
    #     output = output.cpu()
    # output_im = output.numpy()
    # output_im = np.transpose(output_im, (0, 2, 3, 1))
    # output_im = np.squeeze(output_im)
    return output


class Occ_Check_Model():

    def __init__(self, **kwargs):
        self.occ_type = 'for_back_check'
        self.occ_alpha_1 = 0.01
        self.occ_alpha_2 = 0.5
        self.obj_out_all = 'all'  # obj, out, all, when boundary dilated warping is used, this should be 'obj'

        # self.update(kwargs)
        self._check()

    def _check(self):
        assert self.occ_type in ['for_back_check', 'forward_warp']
        assert self.obj_out_all in ['obj', 'out', 'all']

    def __call__(self, flow_f, flow_b):
        '''
        input is optical flow. Using forward-backward checking to compute occlusion regions. 0 stands for occ region. 1 is for other regions.
        '''
        # regions that moving out of the image plane
        if self.obj_out_all == 'out':
            out_occ_fw = self.torch_outgoing_occ_check(flow_f)
            out_occ_bw = self.torch_outgoing_occ_check(flow_b)
            return out_occ_fw, out_occ_bw

        # all occlusion regions
        if self.occ_type == 'for_back_check':
            occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b)
        elif self.occ_type == 'forward_warp':
            raise ValueError('not implemented')
        else:
            raise ValueError('not implemented occlusion check method: %s' % self.occ_type)

        if self.obj_out_all == 'all':
            return occ_1, occ_2

        # 'out' regions are not considered as occlusion
        if self.obj_out_all == 'obj':
            out_occ_fw = self.torch_outgoing_occ_check(flow_f)
            out_occ_bw = self.torch_outgoing_occ_check(flow_b)
            obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
            obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
            return obj_occ_fw, obj_occ_bw

        raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

    def _forward_backward_occ_check(self, flow_fw, flow_bw):
        """
        In this function, the parameter alpha needs to be checked
        # 0 means the occlusion region where the photo loss we should ignore
        """

        def length_sq(x):
            # torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.pow(temp, 0.5)
            # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
            return temp

        mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
        flow_bw_warped = torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
        flow_fw_warped = torch_warp(flow_fw, flow_bw)
        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped
        occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2

        occ_fw = length_sq(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
        occ_bw = length_sq(flow_diff_bw) < occ_thresh
        # if IF_DEBUG:
        #     temp_ = sum_func(flow_diff_fw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
        #     temp_ = sum_func(flow_diff_bw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
        #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
        #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
        return occ_fw.float(), occ_bw.float()

    def _forward_backward_occ_check_unflow(self, flow_fw, flow_bw):
        """
        In this function, the parameter alpha needs to be checked
        # 0 means the occlusion region where the photo loss we should ignore
        """

        flow_bw_warped = torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
        flow_fw_warped = torch_warp(flow_fw, flow_bw)
        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped

        mag_sq_fw = (flow_fw * flow_fw).sum(1, keepdim=True) + (flow_bw_warped * flow_bw_warped).sum(1, keepdim=True)
        occ_thresh_fw = self.occ_alpha_1 * mag_sq_fw + self.occ_alpha_2
        mag_sq_bw = (flow_bw * flow_bw).sum(1, keepdim=True) + (flow_fw_warped * flow_fw_warped).sum(1, keepdim=True)
        occ_thresh_bw = self.occ_alpha_1 * mag_sq_bw + self.occ_alpha_2

        occ_fw = (flow_diff_fw * flow_diff_fw).sum(1, keepdim=True) < occ_thresh_fw  # 0 means the occlusion region where the photo loss we should ignore
        occ_bw = (flow_diff_bw * flow_diff_bw).sum(1, keepdim=True) < occ_thresh_bw

        return occ_fw.float(), occ_bw.float()

    def _forward_warp_occ_check(self, flow_bw):
        raise ValueError('not implemented')

    @classmethod
    def torch_outgoing_occ_check(cls, flow):
        # out going pixels=0, others=1
        B, C, H, W = flow.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        flow_x, flow_y = torch.split(flow, 1, 1)
        if flow.is_cuda:
            xx = xx.cuda()
            yy = yy.cuda()
        # tools.check_tensor(flow_x, 'flow_x')
        # tools.check_tensor(flow_y, 'flow_y')
        # tools.check_tensor(xx, 'xx')
        # tools.check_tensor(yy, 'yy')
        pos_x = xx + flow_x
        pos_y = yy + flow_y
        # tools.check_tensor(pos_x, 'pos_x')
        # tools.check_tensor(pos_y, 'pos_y')
        # print(' ')
        # check mask
        outgoing_mask = torch.ones_like(pos_x)
        outgoing_mask[pos_x > W - 1] = 0
        outgoing_mask[pos_x < 0] = 0
        outgoing_mask[pos_y > H - 1] = 0
        outgoing_mask[pos_y < 0] = 0
        return outgoing_mask.float()

    @classmethod
    def torch_get_obj_occ_check(cls, occ_mask, out_occ):
        outgoing_mask = torch.zeros_like(occ_mask)
        if occ_mask.is_cuda:
            outgoing_mask = outgoing_mask.cuda()
        outgoing_mask[occ_mask == 1] = 1  # not occluded regions =1
        outgoing_mask[out_occ == 0] = 1  # 'out' regions=1, the rest regions=0(object moving inside the image plane)
        return outgoing_mask

