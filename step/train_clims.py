import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
from imutils import visual_debug
from clip_utils import clip_forward,clip_forward_F, CLIPContrastiveLoss,CLIPTripletLoss
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import voc12.dataloader
from misc import pyutils, torchutils
import os, math
from torch import nn
import types
import argparse
import sys
from graph.grm_layer4 import *
from graph import *
from graph.voc_data import *
import net.resnet50_clims
#from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from PIL import Image
import clip
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
#import graph.gcn_layer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from .CAMss import save_final_epoch_visuals_single, load_original_image_tensor
# 定义常量
try:
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


BACKGROUND_CATEGORY = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake",
    "water", "river", "sea", "railway", "railroad", "keyboard", "helmet",
    "cloud", "house", "mountain", "ocean", "road", "rock", "street",
    "valley", "bridge", "sign"
]

new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair seat', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                   ]



#只监督 refined_cams 中高置信度区域，降低噪声干扰
def confident_ce_loss(logits, refined_cams, threshold=0.5):
    # logits: [B, C, H, W], refined_cams: [B, C, H, W]
    probs = torch.sigmoid(logits)
    confident_mask = (refined_cams > threshold).float()

    loss = F.binary_cross_entropy(probs * confident_mask, refined_cams * confident_mask, reduction='sum')
    return loss / (confident_mask.sum() + 1e-6)


#Class-wise Dropout（类级别随机Drop）
#随机“抹除”某些类的区域（如CAM响应区域），迫使模型学习其他未激活的区域，从而提升整体区域的识别能力。
def classwise_dropout(cam_mask, drop_prob=0.3):
    """
    cam_mask: [B, C, H, W] → 类别响应区域，通常来自CAM或预测
    随机将某些类的激活区域mask为0
    """
    B, C, H, W = cam_mask.shape
    drop_mask = torch.ones((B, C, 1, 1), device=cam_mask.device)

    for b in range(B):
        for c in range(C):
            if torch.rand(1).item() < drop_prob:
                drop_mask[b, c] = 0  # mask this class

    cam_mask_dropped = cam_mask * drop_mask  # [B, C, H, W]
    return cam_mask_dropped

#前景类别聚焦 loss
#对激活图中响应较高区域施加约束，鼓励其集中、清晰。
def cam_compactness_loss(x, labels):
    """
    通过最大值和均值之比衡量是否“集中”
    """
    B, C, H, W = x.shape
    loss = 0.0
    for b in range(B):
        for c in range(C):
            if labels[b, c] == 1:
                feat = x[b, c]
                ratio = feat.mean() / (feat.max() + 1e-6)
                loss += ratio
    return loss / B

# 定义辅助函数和类
def reshape_transform(tensor, height=28, width=28):
    #print("##########tensor:",tensor.shape)
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # 将通道移动到第一维度，类似于CNN
    result = result.transpose(2, 3).transpose(1, 2)
    #print("##########result:",result.shape)
    return result


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(n_splits-1)*part:])
    return dataset_list

def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(
            int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), 
            int(np.ceil(scale * int(ori_width) / patch_size) * patch_size)
        )
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs

def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)

def apply_caa(grayscale_cam, attn_weight, aff_mask):
    """
    应用类相关注意力亲和力 (CAA) 方法对 CAM 进行进一步处理。
    
    参数:
    - grayscale_cam: numpy array, 原始 CAM
    - attn_weight: torch.Tensor, 注意力权重
    - aff_mask: torch.Tensor, 亲和掩码
    
    返回:
    - refined_cam: numpy array, 处理后的 CAM
    """
    # 基于注意力权重计算转移矩阵
    trans_mat = attn_weight / torch.sum(attn_weight, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    
    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)
    
    trans_mat = trans_mat * aff_mask
    
    cam_to_refine = torch.FloatTensor(grayscale_cam).view(-1, 1)
    cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(grayscale_cam.shape[0], grayscale_cam.shape[1])
    cam_refined = cam_refined.cpu().numpy().astype(np.float32)
    refined_cam = scale_cam_image([cam_refined], (32, 32))[0]
    
    return refined_cam


def compute_pixel_confusion_matrix(model, val_loader, num_classes=20, device='cuda'):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32).to(device)

    with torch.no_grad():
        for pack in val_loader:
            images = pack['img'].to(device)
            labels = pack['label'].to(device)  # [B, C]
            outputs,_ = model(images)  # [B, C, H, W]

            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            #cams = (outputs > 0).float()  # [B, C, H, W]
            cams = torch.sigmoid(outputs)

            for i in range(images.shape[0]):
                # 获取图像的图像级标签
                gt_classes = torch.nonzero(labels[i]).squeeze(1)

                for cls in gt_classes:
                    mask = cams[i, cls] > 0.5
                    pred_cls = preds[i][mask]  # 当前gt类区域上预测出的类别

                    for pred in torch.unique(pred_cls):
                        confusion[cls, pred] += (pred_cls == pred).sum()

    model.train()
    return confusion

def update_relation_matrix1(relation_matrix, confusion_matrix, alpha=0.0001):
    """
    根据混淆矩阵更新关系矩阵。
    
    参数:
        relation_matrix (torch.Tensor): [20,20] 的关系矩阵。
        confusion_matrix (torch.Tensor): [20,20] 的混淆矩阵。
        learning_rate (float): 学习率。
    
    返回:
        updated_relation_matrix (torch.Tensor): 更新后的关系矩阵。
    """
    # 高混淆意味着需要减少关系权重
    gradient = confusion_matrix.float()  # 将整型转为浮点型
    updated_relation_matrix = relation_matrix - alpha * gradient
    
    # 确保关系矩阵非负
    updated_relation_matrix = torch.clamp(updated_relation_matrix, min=0.0)
    
    # 归一化关系矩阵，使每行的和为1
    row_sum = updated_relation_matrix.sum(dim=1, keepdim=True) + 1e-5
    updated_relation_matrix = updated_relation_matrix / row_sum

    return updated_relation_matrix

def update_relation_matrix(relation_matrix, confusion_matrix, alpha=0.1):
    """
    根据像素级混淆情况更新关系矩阵：
    - 高混淆：降低关系权重；
    - 高正确率：增强权重。
    """
    # 构建目标权重变化矩阵
    mis_confusion = confusion_matrix.clone()
    eye = torch.eye(relation_matrix.size(0)).to(relation_matrix.device)

    # 提取非对角元素作为“负反馈”
    off_diag = mis_confusion * (1 - eye)
    # 提取对角线元素作为“正反馈”
    diag = torch.diag(mis_confusion)

    # 归一化
    diag = diag / (diag.max() + 1e-5)
    off_diag = off_diag / (off_diag.max() + 1e-5)

    delta = torch.zeros_like(relation_matrix)

    for i in range(relation_matrix.size(0)):
        for j in range(relation_matrix.size(1)):
            if i == j:
                delta[i, j] += alpha * diag[i]  # 增强自连
            else:
                delta[i, j] -= alpha * off_diag[i, j]  # 减弱误分类强的边

    # 更新
    updated = relation_matrix + delta

    # 限制范围
    #updated = torch.clamp(updated, 0.001, 1.0)#0.001
    updated = torch.clamp(updated, min=0.0)
    updated = updated / (updated.sum(dim=1, keepdim=True) + 1e-6)

    return updated



def visualize_relation_matrix2(matrix_tensor, epoch, save_dir='./logs/relation_matrix_vis'):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    labels = ['aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train',
                  'tvmonitor']
    os.makedirs(save_dir, exist_ok=True)

    # 移到CPU并转换为numpy
    matrix_tensor = matrix_tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_tensor, cmap='viridis')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Title and color bar
    ax.set_title("Normalized Relation Matrix Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    fig.tight_layout()

    plt.savefig(os.path.join(save_dir, f'relation_matrix_epoch_{epoch+1}.png'))
    plt.close()




def save_relation_matrix_json_simple(relation_matrix, epoch, save_dir="./logs/relation_matrix_json"):
    os.makedirs(save_dir, exist_ok=True)
    matrix_list = relation_matrix.detach().cpu().tolist()  # 转为嵌套 list

    save_path = os.path.join(save_dir, f"relation_matrix_epoch_{epoch+1}.json")
    with open(save_path, 'w') as f:
        json.dump(matrix_list, f, indent=4)  # 仅保存关系矩阵值本身


# ------------------------------
# 验证函数
# ------------------------------
def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss')
    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss': loss.item()})
    model.train()
    print('loss: %.4f' % (val_loss_meter.pop('loss')))
    return


def run(args):
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=20)

    # initialize backbone network with baseline CAM
    model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)
    with open("./graph/CM_kg_57_info.json","rb") as f:
    #with open("./graph/CM_freq_info.json","rb") as f:
        info = json.load(f)
        KF_All_VOC_info = info['KG_VOC_info']
        #KF_All_VOC_info = info['KF_All_VOC_info']
        graph_adj_mat = np.asarray(KF_All_VOC_info['S'])

         # 初始化关系矩阵
    def get_initial_relation_matrix():
        # 从ConceptNet或其他来源获取实际的关系矩阵
        # 这里用随机值作为示例，请替换为实际的关系矩阵
        print('the adj mat is\n',graph_adj_mat)
        relation_matrix = graph_adj_mat
        # 归一化每行
        relation_matrix = relation_matrix / (relation_matrix.sum(axis=1, keepdims=True) + 1e-6)
        return relation_matrix

    initial_relation_matrix = get_initial_relation_matrix()  # [20,20]
    relation_matrix = torch.tensor(initial_relation_matrix, dtype=torch.float32).to('cuda')  # [20,20]

    # 定义FeatureGraphConvolution模块
    feature_gcn = FeatureGraphConvolution(
        relation_matrix=relation_matrix
    ).to('cuda')

    Fusion = AdaptiveGateFusion(in_channels=20).cuda()
   
    feature = DGMN(20,20).cuda()

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': feature_gcn.parameters(), 'lr': 0.5, 'weight_decay': args.cam_weight_decay},#0.5-0.596
        {'params': feature.parameters(), 'lr': 0.05, 'weight_decay': args.cam_weight_decay},#0.5-0.596
        {'params': Fusion.parameters(), 'lr': 5, 'weight_decay': args.cam_weight_decay},#0.5-0.596
       
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()


    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(dname='voc')
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    print(hyper)

    # CLIP
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip, device=device)
   
    clip_model.eval()
     ######################################
    def zeroshot_classifier(classnames, templates, model):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).to(device) #tokenize
                class_embeddings = model.encode_text(texts)#.detach() #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights.t()

    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], clip_model)#['a rendering of a weird {}.'], model)
    fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], clip_model)#['a rendering of a weird {}.'], model)

    def align_refined_cams(refined_cams, label_id_list_all, num_classes=20):
        """
        将每张图片的动态类别 CAM 映射到固定类别维度 [B,C,H,W]
        """
        B = len(refined_cams)
        _, H, W = refined_cams[0].shape
        aligned_cams = torch.zeros((B, num_classes, H, W), device=refined_cams[0].device)

        for b in range(B):
            cams = refined_cams[b]  # [Ni,H,W]
            labels = label_id_list_all[b]  # Ni

            for i, cls_idx in enumerate(labels):
                aligned_cams[b, cls_idx, :, :] = cams[i]

        return aligned_cams

    # 初始化 CLIP 模型用于 CAM 生成
    clip_model_cam, preprocess_cam = clip.load(args.clip, device=device)
    clip_model_cam.eval()
    for param in clip_model_cam.parameters():
        param.requires_grad = False

    target_layers = [clip_model_cam.visual.transformer.resblocks[-1].ln_1]
    cam_generator = GradCAM(
        model=clip_model_cam, 
        target_layers=target_layers, 
        #use_cuda=torch.cuda.is_available(),
        reshape_transform=reshape_transform
    )

    # 生成背景和前景文本特征
    CBLoss = CLIPContrastiveLoss(clip_model, dataset='voc')
    
    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    # transform multi-hot label to class index label
    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()

    # 放在 epoch 循环外
    hyper = [float(h) for h in args.hyper.split(',')]
    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            
            label = pack['label'].cuda(non_blocking=True)

            fg_label = preprocess(label.cpu())

            x, cams = model(img)
            batch_size,C,H,W=x.shape
            
            GCN_1 = feature(x)
            GCN_2 = feature_gcn(x)  # 应用GCN增强特征图 [16,20,32,32]
            x = GCN_1 + GCN_2
            
            loss_cam = cam_compactness_loss(x, label)

            mask_cams = torch.sigmoid(cams).detach() > 0.5  # [B, C, H, W]
            mask_cams = mask_cams.float()
            masked_pseudo_label = classwise_dropout(mask_cams)
            
            loss_class = F.binary_cross_entropy_with_logits(x1, masked_pseudo_label, reduction='mean')

            N, C, H, W = x.size()
            optimizer.zero_grad()

            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()
            
            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 20, 1, clip_input_size,
                                                                                                clip_input_size)
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(label == 1, as_tuple=False)

            
            for j in range(temp_idx.shape[0]):
                #print("##########fg_indices[j]:",fg_indices[j])
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)
            
            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='voc'), 1)
            Loss_O = CBLoss(fg_224_eval, fg_label[fg_indices])
            
            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)
            
            L_REG = torch.mean(x)
            
             # 生成 refined_cam
            ###################################################################
            refined_cams = []
            attn_weight = None
            attn_weight_list = []
            label_id_list_all = []
            grayscalecam_cam_per_image = []
            for i in range(N):
                label_id_list = []
                label_list = []
                refined_cam_per_image = []
                for class_idx in range(C):
                    if label[i, class_idx] == 0:
                        label_id_list.append(class_idx)#label_id_list: [14, 19]
                        label_list.append(new_class_names[class_idx])#label_list: ['person with clothes,people,human', 'tvmonitor screen']
                
                label_id_list_all.append(label_id_list)
                bg_features_temp = bg_text_features  # [bg_id_for_each_image[im_idx]].to(device_id)
                fg_features_temp = fg_text_features[label_id_list]
                text_features_temp = torch.cat([bg_features_temp, fg_features_temp], dim=0)

                img_pil = Image.fromarray(
                        ((img[i].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))
                        )
                input_img = preprocess_cam(img_pil).unsqueeze(0).to(device)
                image_features,attn_weight_list = clip_model.encode_image(input_img,224,224)  # 使用CLIP提取图像特征

                input_cam = [image_features, text_features_temp, 224, 224]
                for idx, lab in enumerate(label_list):
                    
                    target = ClipOutputTarget(label_list.index(lab))#label_list.index(lab): 0， #label_list.index(lab): 1
                    
                    # 生成 CAM
                    grayscale_cam, logits_per_image, attn_weight_last = cam_generator(
                        input_tensor=input_cam, 
                        targets=[target],
                        target_size=None
                    )  # 输出 shape: [1, H_cam, W_cam]
                    
                    grayscale_cam = grayscale_cam[0, :]
                    grayscalecam = scale_cam_image([grayscale_cam], (32, 32))[0]
                    grayscalecam_cam_per_image.append(torch.tensor(grayscalecam).cuda())
                   
                    # 生成亲和掩码 (aff_mask)
                    box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                    aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
                    for i_ in range(cnt):
                        x0_, y0_, x1_, y1_ = box[i_]
                        aff_mask[y0_:y1_, x0_:x1_] = 1
                           
                    aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
                    if idx == 0:
                        attn_weight_list.append(attn_weight_last)
                        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                        attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                        attn_weight = torch.mean(attn_weight, dim=0)
                        attn_weight = attn_weight[0].cpu().detach().float()
                    
                        # 应用 CAA 方法
                    refined_cam = apply_caa(grayscale_cam, attn_weight, aff_mask)
                    refined_cam_per_image.append(torch.tensor(refined_cam).cuda())
                    
                refined_cams.append(torch.stack(refined_cam_per_image,dim=0))  # [20,32,32]

            # 调用示例:
            refined_cams = align_refined_cams(refined_cams, label_id_list_all, num_classes=20)  # [B,20,H,W]
            loss_align = confident_ce_loss(x,refined_cams, threshold=0.6)
            
            
             # 组合所有损失
            loss = hyper[0]*L_OTM + hyper[1]*L_BTM + hyper[2]*L_CBS + hyper[3]*L_REG  + 1*Loss_O + 1*loss_cam + 0.06*loss_class + 0.05*loss_align#0.05-0.60799
    
            loss.backward()
            optimizer.step()
            # 释放内存

            avg_meter.add({
                'loss1': loss.item(),
                'L_OTM': L_OTM.item(),
                'L_BTM': L_BTM.item(),
                'L_CBS': L_CBS.item(),
                'L_REG': L_REG.item(),
                'Loss_O': Loss_O.item(),
                'loss_cam': loss_cam.item(),
                'loss_class': loss_class.item(),
                'loss_align': loss_align.item()
            })

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'L_OTM:%.4f' % (avg_meter.pop('L_OTM')),
                      'L_BTM:%.4f' % (avg_meter.pop('L_BTM')),
                      'L_CBS:%.4f' % (avg_meter.pop('L_CBS')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'Loss_O:%.4f' % (avg_meter.pop('Loss_O')),
                      'loss_cam:%.4f' % (avg_meter.pop('loss_cam')),
                      'loss_class:%.4f' % (avg_meter.pop('loss_class')),
                      'loss_align:%.4f' % (avg_meter.pop('loss_align')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # visualize class activation maps during training if needed.
                # visual_debug(img, label, x, 'vis/clims_v2_voc12_cam_vis', optimizer.global_step, num_classes=21,
                #             dataset='coco', phase='train')

        # 在每个epoch结束后，计算混淆矩阵并更新关系矩阵
        # ✅ 在 epoch 循环内，每轮训练后调用
        confusion_matrix = compute_pixel_confusion_matrix(model, val_data_loader, num_classes=20, device=device)
        relation_matrix = update_relation_matrix1(relation_matrix, confusion_matrix, alpha=0.05)#0.05
       
        # 可视化 + 保存
        #visualize_relation_matrix2(relation_matrix, epoch=ep)
        #save_relation_matrix_json_simple(relation_matrix, ep)
        
        feature_gcn.graph_reasoning1.graph_norm_adj = normalize_adjacency(relation_matrix)
        feature_gcn.graph_reasoning2.graph_norm_adj = normalize_adjacency(relation_matrix)
        
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    torch.cuda.empty_cache()
