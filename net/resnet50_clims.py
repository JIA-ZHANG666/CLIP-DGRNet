import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50_v2 as resnet50


class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class Classifier_Module1(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class Classifier_Module(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(Classifier_Module, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print("#######x",x.shape)
        return sum([stage(x) for stage in self.children()])

class CLIMS(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)
        

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

        #self.g_conv = nn.Conv2d(20,20,1,bias=False)
        #self.conv1_add = nn.ModuleList([self.g_conv])
    # -------------------------------------------------------------
    # For DDP & syncBN training, must set 'requires_grad = False' before passing to DDP
    # https://discuss.pytorch.org/t/how-does-distributeddataparallel-handle-parameters-whose-requires-grad-flag-is-false/90736/1
        self._freeze_layers()

    def _freeze_layers(self):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
    # --------------------------------------------------------------

    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        #x1=x
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)
        #print("#######x",x.shape)

        x1 = self.classifier(x)
        #logist = torchutils.gap2d(x, keepdims=True)
        #logist = logist.view(-1, self.n_classes)
        #g_img = self.g_conv1(x1)
        cams = F.conv2d(x, self.classifier.weight)
        cams = F.relu(cams)
        #print("#####cams:",cams.shape)
        cams = cams / (F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2) * x.unsqueeze(1)  # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)
        cams_feature = torch.mean(cams_feature, 2).reshape(16,20,32,32)
        #print("#####cams_feature:",cams_feature.shape)
        return torch.sigmoid(x1) ,cams_feature #,logist#, torch.sigmoid(g_img)

    def train(self, mode=True):
        super(CLIMS, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class Net_CAM(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)

        return x, cams, feature


class Net_CAM_Feature(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM_Feature, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)  # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams / (F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2) * feature.unsqueeze(1)  # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)
        cams_feature = torch.mean(cams_feature, -1)

        return x, cams_feature, cams


class CAM(CLIMS):

    def __init__(self, stride=16, n_classes=20):
        super(CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward1(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight * self.classifier.weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()

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

class Class_Predictor1(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_classes, -1)  # bs*20*2048
        mask = label > 0  # bs*20

        feature_list = [x[i][mask[i]] for i in range(batch_size)]  # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            print("#######prediction:",logit.shape)
            if label.shape[0] == 0:
                continue
            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        return loss / batch_size


class Class_Predictor2(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(representation_size, num_classes, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_classes, -1)  # bs * 20 * 256
        mask = label > 0  # bs * 20

        feature_list = [x[i][mask[i]] for i in range(batch_size)]  # bs * n * 256
        prediction = [self.classifier(y) for y in feature_list]  # list of tensors of shape [n, num_classes]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]  # list of tensors of shape [n]

        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label).sum().float()
            num += label.size(0)

        return loss / batch_size


class SemanticContextLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(SemanticContextLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.semantic_centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

    def forward(self, semantic_nodes, labels):
        # semantic_nodes: [batch_size, num_nodes, embedding_dim]
        # labels: [batch_size, num_classes]
        
        batch_size = semantic_nodes.size(0)
        num_nodes = semantic_nodes.size(1)
        
        # Expand semantic centers to match the batch size
        semantic_centers_expanded = self.semantic_centers.expand(batch_size, -1, -1)  # [batch_size, num_classes, embedding_dim]
        
        # Compute similarity scores vi for each node and class
        similarity_scores = torch.bmm(semantic_nodes, semantic_centers_expanded.transpose(1, 2))  # [batch_size, num_nodes, num_classes]
        similarity_scores = self.sigmoid(similarity_scores)  # Apply sigmoid activation
        
        # Flatten similarity scores and labels for BCE loss
        similarity_scores_flat = similarity_scores.view(batch_size * num_nodes, self.num_classes)  # [batch_size * num_nodes, num_classes]
        labels_flat = labels.unsqueeze(1).expand(batch_size, num_nodes, self.num_classes)
        labels_flat = labels_flat.contiguous().view(batch_size * num_nodes, self.num_classes)  # [batch_size * num_nodes, num_classes]
        
        # Compute SC loss using BCE loss
        sc_loss = self.bce_loss(similarity_scores_flat, labels_flat)
        
        return sc_loss

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)