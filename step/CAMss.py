import os
import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

import torchvision.transforms as transforms

def load_original_image_tensor(img_name, voc12_root):
    """
    加载 VOC 原图为张量形式，供 CAM 可视化使用
    Args:
        img_name: 图像名，如 "2007_000926"
        voc12_root: VOC 数据集根路径
    Returns:
        torch.FloatTensor, shape: [3, H, W], range: [0, 1]
    """
    img_path = os.path.join(voc12_root, "JPEGImages", img_name + ".jpg")
    img = Image.open(img_path).convert("RGB")
    transform = transforms.ToTensor()  # 自动转为 [0,1] float32 tensor 且为 [3, H, W]
    return transform(img)

def normalize_to_uint8(x):
    """Normalize tensor or ndarray to [0, 255] uint8"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)

def save_overlay_cam(cam_map, original_img_tensor, save_path, alpha=0.8):
    """
    cam_map: [H, W] uint8
    original_img_tensor: [3, H, W] torch.Tensor, float32, [0,1]
    """
    cam_color = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

    ori_img = original_img_tensor.cpu().numpy()
    ori_img = np.transpose(ori_img, (1, 2, 0))  # [H, W, 3]
    ori_img = (ori_img * 255).astype(np.uint8)  # 原图已在 [0,1] 区间，无需再归一化

    cam_color = cv2.resize(cam_color, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(ori_img, 1 - alpha, cam_color, alpha, 0)
    Image.fromarray(blended).save(save_path)

def save_final_epoch_visuals_single(x_i, img_i, label_i, grayscale_cam_i, refined_cam_i,
                                     img_name='sample',
                                     save_dir='./logs/final_epoch_cam',
                                     refined_fuse_mode='mean',
                                     original_size=None,
                                     original_img_tensor=None):
    """
    保存单张图像的全部可视化输出
    Args:
        x_i: [C, H_feat, W_feat]
        img_i: [3, H_crop, W_crop] 裁剪后图像，仅用于保存 cropped_input.png
        label_i: [C]
        grayscale_cam_i: [H_feat, W_feat]
        refined_cam_i: [N_fg, H_feat, W_feat] or [H_feat, W_feat]
        original_size: tuple(int, int), e.g., (H_ori, W_ori)
        original_img_tensor: [3, H, W] float32 in [0,1], 用于 overlay CAM（优先使用）
    """
    os.makedirs(save_dir, exist_ok=True)
    C, H_feat, W_feat = x_i.shape

    _, H_default, W_default = img_i.shape
    H_ori, W_ori = original_size if original_size is not None else (H_default, W_default)

    # 保存训练裁剪图像
    img_np = img_i.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    Image.fromarray((img_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f"{img_name}_cropped_input.png")
    )

    # 如果没有传入原图张量，则 fallback 为训练图像
    overlay_img = original_img_tensor if original_img_tensor is not None else img_i

    # x CAM
    fg_indices = (label_i == 1).nonzero(as_tuple=False).squeeze()
    if fg_indices.numel() > 0:
        x_fg = x_i[fg_indices]
        if x_fg.ndim == 2:
            x_fg = x_fg.unsqueeze(0)
        x_combined = x_fg.mean(0) if refined_fuse_mode == 'mean' else x_fg.max(0)[0]

        x_resized = F.interpolate(x_combined.unsqueeze(0).unsqueeze(0),
                                  size=(H_ori, W_ori), mode='bilinear', align_corners=False).squeeze()
        x_uint8 = normalize_to_uint8(x_resized)
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_x_fgcam.png"),
                    cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET))
        save_overlay_cam(x_uint8, overlay_img, os.path.join(save_dir, f"{img_name}_x_fgcam_overlay.png"))

    # grayscale CAM
    """if isinstance(grayscale_cam_i, np.ndarray):
        grayscale_cam_i = torch.from_numpy(grayscale_cam_i)
    gray_resized = F.interpolate(grayscale_cam_i.unsqueeze(0).unsqueeze(0),
                                 size=(H_ori, W_ori), mode='bilinear', align_corners=False).squeeze()
    gray_uint8 = normalize_to_uint8(gray_resized)
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_grayscalecam.png"),
                cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET))
    save_overlay_cam(gray_uint8, overlay_img, os.path.join(save_dir, f"{img_name}_grayscalecam_overlay.png"))"""

    # grayscale CAM
    if grayscale_cam_i.ndim == 3:
        grayscale_combined = grayscale_cam_i.mean(0) if refined_fuse_mode == 'mean' else grayscale_cam_i.max(0)[0]
    else:
        grayscale_combined = grayscale_cam_i
    grayscale_resized = F.interpolate(grayscale_combined.unsqueeze(0).unsqueeze(0),
                                    size=(H_ori, W_ori), mode='bilinear', align_corners=False).squeeze()
    grayscale_uint8 = normalize_to_uint8(grayscale_resized)
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_grayscalecam.png"),
                cv2.applyColorMap(grayscale_uint8, cv2.COLORMAP_JET))
    save_overlay_cam(grayscale_uint8, overlay_img, os.path.join(save_dir, f"{img_name}_grayscalecam_overlay.png"))

    # refined CAM
    if refined_cam_i.ndim == 3:
        refined_combined = refined_cam_i.mean(0) if refined_fuse_mode == 'mean' else refined_cam_i.max(0)[0]
    else:
        refined_combined = refined_cam_i
    refined_resized = F.interpolate(refined_combined.unsqueeze(0).unsqueeze(0),
                                    size=(H_ori, W_ori), mode='bilinear', align_corners=False).squeeze()
    refined_uint8 = normalize_to_uint8(refined_resized)
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_refined_cam.png"),
                cv2.applyColorMap(refined_uint8, cv2.COLORMAP_JET))
    save_overlay_cam(refined_uint8, overlay_img, os.path.join(save_dir, f"{img_name}_refined_cam_overlay.png"))
