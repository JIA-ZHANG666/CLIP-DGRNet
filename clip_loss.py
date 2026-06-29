import torch
import clip
from clip_utils import background_dict, category_dict

# maximize similarity
class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        #print("#########weight2:",weights)
        return -(torch.log(x + self.margin) * weights).mean()

# minimize similarity
class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        #print("#########weight2:",weights)
        return -(torch.log(1 - x + self.margin) * weights).mean()

# suppress background activation
class BackgroundSuppressionLoss(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self, threshold=0.26, dname='coco'):
        super(BackgroundSuppressionLoss, self).__init__()
        self.dname = dname
        self.background = background_dict[dname]
        self.threshold = threshold
        print(f'Use CBSLoss! threshold: {threshold}')

    def forward(self, clip_model, images, eps=0.0001):
        image_features,_ = clip_model.encode_image(images,224,224)  # [N1, C]
        #image_features.requires_grad = True  # 允许梯度
        x = image_features.permute(1, 0, 2)  # LND -> NLD
        #x = clip_model.visual.ln_post(x)
        #image_features = torch.mean(x[:,1:,:],dim=1)
        image_features = clip_model.visual.ln_post(x[:, 0, :])
        #print("#########image_featuresSS:",image_features.shape)

        # ---- 关键：如果视觉侧有 proj，就把 768 -> 512 ----
        if clip_model.visual.proj is not None:
            # 一般 CLIP 模型都会有 self.visual.proj
            image_features = image_features @ clip_model.visual.proj
        #print("#########image_featuresSS:",image_features.shape)
        text_features = clip_model.encode_text(clip.tokenize(self.background).cuda())  # [N2, C]

        # normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (image_features @ text_features.t())  # [N1, N2]
        mask = torch.zeros_like(logits_per_image)
        mask = torch.where(logits_per_image > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        return -(torch.log(1 - logits_per_image) * mask).sum()


import torch
import torch.nn as nn

class BackgroundSuppressionLoss2(nn.Module):
    def __init__(self, dname='voc'):
        super(BackgroundSuppressionLoss, self).__init__()
        # 根据数据集名称初始化必要的参数
    
    def forward(self, clip_model, img, bg_text_features, fg_text_features, label):
        """
        计算背景抑制损失
        
        参数:
        - clip_model: CLIP 模型
        - img: torch.Tensor, 输入图像
        - bg_text_features: torch.Tensor, 背景文本特征
        - fg_text_features: torch.Tensor, 前景文本特征
        - label: torch.Tensor, 标签
        
        返回:
        - loss: torch.Tensor, 背景抑制损失
        """
        # 编码图像
        image_features = clip_model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 计算前景相似度
        fg_sim = image_features @ fg_text_features.t()
        # 计算背景相似度
        bg_sim = image_features @ bg_text_features.t()
        
        # 计算损失：前景相似度最大化，背景相似度最小化
        loss_fg = -torch.log(torch.sigmoid(fg_sim).clamp(min=1e-6))
        loss_bg = -torch.log(1 - torch.sigmoid(bg_sim).clamp(min=1e-6))
        
        # 只对存在标签的前景类别计算损失
        loss = (loss_fg + loss_bg).sum()
        return loss