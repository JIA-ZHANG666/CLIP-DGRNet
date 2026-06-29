""""
Define a generic GRM layer model
"""
from pickletools import decimalnl_short
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.coco_data import *
from omegaconf import OmegaConf
#from config.global_settings import GPU_ID
from torch.autograd import Variable
from torch.nn import Parameter
import math
from .graph_util import *
from timm.models.layers import DropPath
import random
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head, num_samples):
        super().__init__()
        height, width = fmap_size
        scale = dim_head ** -0.5
        self.num_samples = num_samples
        self.rel_height = nn.Parameter(torch.randn(height + num_samples - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width + num_samples - 1, dim_head) * scale)

    def rel_to_abs(self, x):
        b, h, l, c = x.shape
        x = torch.cat((x, torch.zeros((b, h, l, 1), dtype=x.dtype, device=x.device)), dim=3)
        x = x.reshape(b, h, l * (c + 1))
        x = torch.cat((x, torch.zeros((b, h, self.num_samples - 1), dtype=x.dtype, device=x.device)), dim=2)
        x = x.reshape(b, h, l + 1, self.num_samples + l - 1)
        x = x[:, :, :l, (l - 1):]
        return x

    def relative_logits_1d(self, q, rel_k):
        logits = torch.matmul(q, rel_k.transpose(0, 1))
        b, h, x, y, r = logits.shape
        logits = logits.reshape(b, h * x, y, r)
        logits = self.rel_to_abs(logits)
        return logits

    def forward(self, q, H, W):
        rel_width = F.interpolate(
            self.rel_width.unsqueeze(0).unsqueeze(0),
            size=(W + 9 - 1, self.rel_width.shape[1]), mode="bilinear").squeeze(0).squeeze(0)
        rel_height = F.interpolate(
            self.rel_height.unsqueeze(0).unsqueeze(0),
            size=(H + 9 - 1, self.rel_height.shape[1]), mode="bilinear").squeeze(0).squeeze(0)

        rel_logits_w = self.relative_logits_1d(q, rel_width)
        q = q.transpose(2, 3)
        rel_logits_h = self.relative_logits_1d(q, rel_height)

        rel_logits_h = F.interpolate(
            rel_logits_h.permute(0, 3, 1, 2),
            size=rel_logits_w.shape[1:3], mode="bilinear").permute(0, 2, 3, 1)

        return rel_logits_w + rel_logits_h

class DGMN2Attention(nn.Module):
    def __init__(self, dim, num_heads=8, fea_size=(32, 32), qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Sample
        self.num_samples = 9
        self.conv_offset = nn.Linear(self.head_dim, self.num_samples * 2, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=3, padding=1)

        # Relative position
        self.pos_emb = RelPosEmb(fea_size, self.head_dim, self.num_samples)

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        offset = self.conv_offset(x.reshape(B, N, self.num_heads, self.head_dim)).permute(0, 2, 3, 1).reshape(B * self.num_heads, self.num_samples * 2, H, W)
        k = k.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        v = v.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        k = self.unfold(k).transpose(1, 2).reshape(B, self.num_heads, N, self.head_dim, self.num_samples)
        v = self.unfold(v).reshape(B, self.num_heads, self.head_dim, self.num_samples, N).permute(0, 1, 4, 3, 2)

        attn = torch.matmul(q.unsqueeze(3), k) * self.scale
        attn_pos = self.pos_emb(q.reshape(B * self.num_heads, 1, H, W, self.head_dim), H, W).reshape(B, self.num_heads, N, 1, self.num_samples)
        attn = attn + attn_pos

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DGMN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DGMN, self).__init__()
        self.gcn1 = GraphConvolution(20, 128)
        self.gcn2 = GraphConvolution(128, 20)
        self.attn = DGMN2Attention(
            dim=20, num_heads=4, fea_size=(32, 32), qkv_bias=True, qk_scale=None, attn_drop=0.1,
            proj_drop=0.1
        )
        self.local_global_relation = LocalGlobalRelationMatrix(feature_dim=1024)
        self.gtn_layer = GTN(num_channels=1024, num_layers=1).cuda()
        self.conv1 = nn.Conv2d(20 +20, 1,
                              kernel_size=1, stride=1)
        

    def compute_compat_batch(self, batch_input, batch_evolve):
        # batch_input [H, W, Dl]
        # batch_evolve [M, Dc]
        # [H, W, Dl] => [H * W, Dl] => [H*W, M, Dl]
        H = batch_input.shape[0]
        W = batch_input.shape[1]
        M = batch_evolve.shape[0]
        Dl = batch_input.shape[-1]
        batch_input = batch_input.reshape( H * W, Dl)
        batch_input = batch_input.unsqueeze(1).repeat([1,M,1])
        # [M,Dc] => [H*W, M, Dc]
        batch_evolve = batch_evolve.unsqueeze(0).repeat([H*W, 1, 1])
        # [H*W, M, Dc+Dl] 
        batch_concat = torch.cat([batch_input, batch_evolve], axis=-1)
        # [H*W, M, Dc+Dl] =>[1,H*W, M, Dc+Dl]
        batch_concat = batch_concat[np.newaxis,:,:,:]
        # [H*W, M, Dc+Dl] =>[1,Dc+Dl,H*W, M]
        batch_concat = batch_concat.transpose(2,3).transpose(1,2)
        #print("@@@@@@@batch_concat",batch_concat.size())
        #[1,Dc+Dl,H*W, M] =>[1,1,H*W, M]
        mapping = self.conv1(batch_concat)
        #[1,1,H*W, M] => [1, H*W, M, 1]
        mapping = mapping.transpose(1,2).transpose(2,3)
        #[1,1,H*W, M] => [H*W, M, 1]
        mapping = mapping.view(-1,mapping.size(2),mapping.size(3))
        #[H*W, M,1] => [H*W, M]
        mapping = mapping.view(mapping.size(0), -1)
        mapping = F.softmax(mapping, dim=0)
        return  mapping

    def forward(self, x):
        b, c, h, w = x.size()
        x2 = x
        adj = self.attn(x2.view(b,c,-1).permute(0, 2, 1),h,w)
        x2 = (F.relu(adj.transpose(2,1).view(b,c,h,w)) * x2).view(b,c,-1).permute(0, 2, 1)
        adj = self.local_global_relation(x2)
        batch_adj = []
        for i in range(adj.size(0)):
            #print("###adj_batch",adj[i].shape)
            adj1 = self.gtn_layer(adj[i])
            batch_adj.append(adj1)
        adj = torch.stack(batch_adj, dim=0)
        adj = normal(adj)
        batch_list = []
        for index in range(x.size(0)):
            batch = self.gcn1(x2[index], adj[index])
            batch = F.relu(batch)
            batch = F.dropout(self.gcn2(batch, adj[index]),0.3)
            batch = F.relu(batch)
            batch_list.append(batch)
        # [?, M, H*W]
        evolved_feat = torch.stack(batch_list, dim=0).transpose(2,1).view(b,c,h,w)
        sp_embedding = evolved_feat

        return sp_embedding

def normal(adj):
    b,n,n = adj.shape
    I = torch.eye(n).unsqueeze(0).cuda()
    adj = adj + I
    d = adj.sum(-1)
    d = torch.pow(d,-0.5)
    D = adj.detach().clone()
    for i in range(adj.size(0)):
        D[i] = torch.diag(d[i])
    norm_A = D.bmm(adj).bmm(D)

    return norm_A


class LocalGlobalRelationMatrix(nn.Module):
    def __init__(self, feature_dim, lambda_local=0.6, lambda_high_order=0.3):
        """
        :param feature_dim: 输入特征维度 (1024)
        :param lambda_local: 控制局部信息 (X'X^T) 和全局信息 (基于余弦相似度的 A_global) 的比例
        :param lambda_high_order: 二阶关系的加权系数 (A^2)
        """
        super(LocalGlobalRelationMatrix, self).__init__()
        self.lambda_local = lambda_local
        self.lambda_high_order = lambda_high_order

    def forward(self, x):
        """
        :param x: 视觉特征, 形状 (B, 20, 1024)
        :return: 关系矩阵 A, 形状 (B, 1024, 1024)
        """
        #B, C, N = x.shape
        #x = x.permute(0, 2, 1)  # (B, 1024, 20)

        # 计算全局相似度
        x_norm = F.normalize(x, p=2, dim=-1)
        A_global = torch.matmul(x_norm, x_norm.transpose(1, 2))  # (B, 1024, 1024)

        # 计算局部信息: 直接使用X'X^T
        A_local = torch.matmul(x, x.transpose(1, 2))

        # 归一化
        A_local = A_local / (torch.sum(A_local, dim=-1, keepdim=True) + 1e-6)
        A_global = A_global / (torch.sum(A_global, dim=-1, keepdim=True) + 1e-6)

        A = self.lambda_local * A_local + (1 - self.lambda_local) * A_global

        # 高阶依赖增强
        A_2 = torch.matmul(A, A)  # A^2
        A_3 = torch.matmul(A_2, A)  # A^3
        A = A + self.lambda_high_order * A_2 + (self.lambda_high_order / 2) * A_3

        return A


class GTN(nn.Module):
    """
    Graph Transformer Network: 结合多个 GT 层，并执行图卷积
    """
    def __init__(self, num_channels, num_layers):
        super(GTN, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        #self.num_nodes = num_nodes

        # 多层 Graph Transformer 层
        # 多层 Graph Transformer 层
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(1, num_channels, first=True))
            else:
                layers.append(GTLayer(num_channels, num_channels, first=False))
        self.layers = nn.ModuleList(layers)


    def forward(self, A):
        """
        A: (batch_size, num_nodes, num_nodes)
        X: (batch_size, num_nodes, w_in)
        """
        for i in range(self.num_layers):
            if i == 0:
                A = self.layers[i](A)
            else:
                A = self.layers[i](A, H_=A)

        return A


class GTLayer(nn.Module):
    """
    Graph Transformer Layer: 生成新的图结构并计算转换后的邻接矩阵
    """
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.first = first
        #self.num_nodes = num_nodes
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        """
        A: (batch_size, num_nodes, num_nodes)
        """
        if self.first:
            result_A = self.conv1(A).unsqueeze(1)
            result_B = self.conv2(A).unsqueeze(0)
        else:
            result_A = H_
            result_B = self.conv1(A).unsqueeze(0)

        H = torch.matmul(result_A, result_B)

        return H

class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,out_channels))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A.cuda()*F.softmax(self.weight, dim=1).cuda(), dim=1)
        
        return A



 
#Graph Reasoning Module
#Graph convolution
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, graph_norm_adj):
        support = torch.matmul(input, self.weight.cuda())
        graph_norm_adj=graph_norm_adj.to(input.cuda())
        output = torch.matmul(graph_norm_adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# 方法一：自适应门控融合
class AdaptiveGateFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels, 1),
            
        )
        
    def forward(self, spatial_feat, semantic_feat):
        # spatial_feat: [B,C,H,W], semantic_feat: [B,C,H,W]
        combined = torch.cat([spatial_feat, semantic_feat], dim=1)
        gate = torch.sigmoid(self.gate_conv(combined))
        fused = gate * spatial_feat + (1 - gate) * semantic_feat

        return fused
    

class FeatureGraphConvolution(nn.Module):
    def __init__(self,  relation_matrix,reduction=16):
        super(FeatureGraphConvolution, self).__init__()


        self.graph_reasoning1 = GraphConvolution(1024,128)
        self.graph_reasoning2 = GraphConvolution(128,128)

        self.graph_adj_mat = relation_matrix
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(128, 20)

    
    def forward(self, x):
        #[？，M, H*W]
        visual_feat = x
        batch, C, H, W = x.size()
        # 聚合空间特征，形成类别特征向量
        class_features = x.view(batch, C, -1)  # [batch, num_classes, in_channels]
        fasttest_embeddings = class_features
        graph_adj = self.graph_adj_mat
        
        graph_norm_adj = normalize_adjacency(graph_adj)
        batch_list = []
        for index in range(visual_feat.size(0)):
            batch1 = self.graph_reasoning1(fasttest_embeddings[index], graph_norm_adj)
            batch1 = F.relu(batch1)
            batch_list.append(batch1)
        # [?, M, H*W]
        evolved_feat = torch.stack(batch_list, dim=0)
        batch_list1 = []
        for index in range(evolved_feat.size(0)):
            evolved_feats = F.dropout(evolved_feat[index], 0.3)
            batch2 = self.graph_reasoning2(evolved_feats, graph_norm_adj)
            batch2 = F.relu(batch2)
            batch_list1.append(batch2)
        # [?, M, H*W]
        kg_embedding = torch.stack(batch_list1, dim=0)#kg_embedding: torch.Size([16, 20, 128])
        kg_embedding = self.fc(kg_embedding)#kg_embedding2: torch.Size([16, 20, 20])
        
        attention_weights = torch.sigmoid(torch.einsum('bchw,bnc->bhwn', x, kg_embedding))  # [B, H, W, num_classes]#attention_weights: torch.Size([16, 32, 32, 20])
        
        # Mean pooling the attention weights across the class dimension
        attention_weights = attention_weights.mean(dim=3)  # [B, H, W]
        
        # Expand attention weights to match the feature dimensions
        attention_weights = attention_weights.unsqueeze(1)  # [B, 1, H, W]
        enhanced_features = x * attention_weights
      
        
        return enhanced_features






if __name__ == "__main__":
   pass