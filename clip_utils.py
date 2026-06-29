import torch
import torch.nn.functional as F

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            'dog',
            'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'],
    #'voc': ['aeroplane', 'bicycle', 'birdavian', 'boat', 'bottle',
                   #'bus', 'car', 'cat', 'chairseat', 'cow',
                  # 'diningtable', 'dog', 'horse', 'motorbike', 'personwithclothespeoplehuman',
                  # 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitorscreen',
                  # ],
    'coco': ['player', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
}

background_dict = {
    'voc': ['a photo of tree.', 'a photo of river.',
            'a photo of sea.', 'a photo of lake.', 'a photo of water.',
            'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
            'a photo of stone.', 'a photo of rocks.'],
    #'voc': ['a photo of ground','a photo of land','a photo of grass','a photo of tree','a photo of building','a photo of wall','a photo of sky','a photo of lake','a photo of water','a photo of river','a photo of sea','a photo of railway','a photo of railroad','a photo of keyboard','a photo of helmet',
                       # 'a photo of cloud','a photo of house','a photo of mountain','a photo of ocean','a photo of road','a photo of rock','a photo of street','a photo of valley','a photo of bridge','a photo of sign',
                       # ]   
    'coco': ['a photo of street sign.', 'a photo of mountain.', 'a photo of video game.', 'a photo of men.',
             'a photo of track.', 'a photo of bus stop.', 'a photo of cabinet.', 'a photo of tray.',
             'a photo of plate.', 'a photo of shirt.', 'a photo of city street.', 'a photo of runway.',
             'a photo of tower.', 'a photo of ramp.', 'a photo of grass.', 'a photo of pillow.',
             'a photo of urinal.', 'a photo of lake.', 'a photo of brick.', 'a photo of fence.',
             'a photo of shower.', 'a photo of airport.', 'a photo of animal.', 'a photo of shower curtain.',
             'a photo of road.', 'a photo of mirror.', 'a photo of jacket.', 'a photo of church.', 'a photo of snow.',
             'a photo of fruit.', 'a photo of hay.', 'a photo of floor.', 'a photo of field.', 'a photo of street.',
             'a photo of mouth.', 'a photo of steam engine.', 'a photo of cheese.', 'a photo of river.',
             'a photo of tree branch.', 'a photo of suit.', 'a photo of child.', 'a photo of soup.', 'a photo of desk.',
             'a photo of tub.', 'a photo of tennis court.', 'a photo of teeth.', 'a photo of bridge.',
             'a photo of sky.', 'a photo of officer.', 'a photo of sidewalk.', 'a photo of dock.',
             'a photo of tree.', 'a photo of court.', 'a photo of rock.', 'a photo of board.',
             'a photo of branch.', 'a photo of pan.', 'a photo of box.', 'a photo of body.',
             'a photo of salad.', 'a photo of dirt.', 'a photo of leaf.', 'a photo of hand.',
             'a photo of highway.', 'a photo of vegetable.', 'a photo of computer monitor.',
             'a photo of door.', 'a photo of meat.', 'a photo of pair.', 'a photo of beach.',
             'a photo of harbor.', 'a photo of ocean.', 'a photo of baseball player.', 'a photo of girl.',
             'a photo of market.', 'a photo of window.', 'a photo of blanket.', 'a photo of boy.', 'a photo of woman.',
             'a photo of bat.', 'a photo of baby.', 'a photo of flower.', 'a photo of wall.', 'a photo of bath tub.',
             'a photo of tarmac.', 'a photo of tennis ball.', 'a photo of roll.', 'a photo of park.'],
}

prompt_dict = ['a photo of {}.']
#a photo is a
#prompt_dict = ['a clean origami {}.']


def to_text(labels, dataset='voc'):
    _d = category_dict[dataset]

    text = []
    #########labels.size:([22, 20])
    #print("########labels.size(0):",labels.size(0))
    for i in range(labels.size(0)):
        idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
        if torch.sum(labels[i]) == 1:
            idx = idx.unsqueeze(0)
        cnt = idx.shape[0] - 1
        if cnt == -1:
            text.append('background')
        elif cnt == 0:
            text.append(prompt_dict[cnt].format(_d[idx[0]]))
        elif cnt == 1:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]]))
        elif cnt == 2:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]]))
        elif cnt == 3:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]]))
        elif cnt == 4:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]]))
        else:
            raise NotImplementedError
    #print("########text:",len(text))-22
    #print("########text:",text)
    return text


import clip

class CLIPTripletLoss(torch.nn.Module):
    def __init__(self, clip_model, dataset='voc', margin=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.dataset = dataset
        self.margin = margin
        self.category_names = category_dict[dataset]

    def forward(self, images, labels,relation_matrix):

        #print("#########labels:",labels.shape)
        _, _,H,W = images.shape
        all_class_names = [f'a photo of a {cls}' for cls in self.category_names]
        class_texts = clip.tokenize(all_class_names).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(class_texts)  # [20, D]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            relation_matrix = relation_matrix.to(text_features.dtype)
            text_features = torch.mm(relation_matrix,text_features) + text_features
            #print("#########text_features:",text_features.shape)

        image_features,_ = self.clip_model.encode_image(images,H,W)
        x = image_features.permute(1, 0, 2)  # LND -> NLDx: torch.Size([27, 50, 768])
        image_features = self.clip_model.visual.ln_post(x[:, 0, :])

        # ---- 关键：如果视觉侧有 proj，就把 768 -> 512 ----
        if getattr(self.clip_model.visual, "proj", None) is not None:
            # 一般 CLIP 模型都会有 self.visual.proj
            image_features = image_features @ self.clip_model.visual.proj
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #image_features = F.normalize(self.clip_model.encode_image(images), dim=1)

        loss = 0
        for i in range(images.size(0)):
            pos_idx = labels[i].argmax()
            neg_indices = (labels[i] == 0).nonzero(as_tuple=False).squeeze()

            anchor = image_features[i]
            positive = text_features[pos_idx]
            negatives = text_features[neg_indices]  # 多个负例

            d_ap = 1 - (anchor @ positive)  # cosine距离
            d_an = 1 - (anchor @ negatives.T)  # 多个负例

            triplet = F.relu(d_ap.unsqueeze(0) - d_an + self.margin)
            loss += triplet.mean()
        return loss / images.size(0)

class CLIPContrastiveLoss(torch.nn.Module):
    def __init__(self, clip_model, dataset='voc', temperature=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.dataset = dataset
        self.temperature = temperature
        self.category_names = category_dict[dataset]

    def forward(self, images, labels):
        # images: [N, 3, 224, 224]
        # labels: [N, 20] one-hot
        _, _,H,W = images.shape
        # Text prompt for all categories
        all_class_names = [f'a photo of a {cls}' for cls in self.category_names]
        class_texts = clip.tokenize(all_class_names).cuda()  # [20, 77]
        with torch.no_grad():
            text_features = self.clip_model.encode_text(class_texts)  # [20, D]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #text_features = F.normalize(text_features, dim=1)

        # Encode images
        image_features,_ = self.clip_model.encode_image(images,H,W)
        x = image_features.permute(1, 0, 2)  # LND -> NLDx: torch.Size([27, 50, 768])
        image_features = self.clip_model.visual.ln_post(x[:, 0, :])

        # ---- 关键：如果视觉侧有 proj，就把 768 -> 512 ----
        if getattr(self.clip_model.visual, "proj", None) is not None:
            # 一般 CLIP 模型都会有 self.visual.proj
            image_features = image_features @ self.clip_model.visual.proj
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        #image_features = self.clip_model.encode_image(images)  # [N, D]
        #image_features = F.normalize(image_features, dim=1)

        # Compute similarities: [N, 20]
        logits = image_features @ text_features.T  # cosine similarity
        logits = logits / self.temperature #logist: torch.Size([26, 20])
        #print("#########logist:",logits.shape)

        # Create target: for each row i, positive label index where labels[i, c] == 1
        #print("#########labels:",labels.shape) #labels: torch.Size([26, 20])
        targets = labels.argmax(dim=1)  # [N] → index of positive class
        #print("#########targets:",targets.shape) #targets: torch.Size([26])

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        return loss

def clip_forward11(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)
    #print("########texts:",len(texts))
   # print("########texts:",texts)
    texts = clip.tokenize(texts).cuda()#([26, 77])

    #print("########texts:",texts.size())

    image_features = clip_model.encode_image(images)#([26, 512])
    text_features = clip_model.encode_text(texts)#([26, 512])
    #print("########text_features:",text_features.size())

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)#([26, 1, 512])
    #print("########image_features:",image_features.size())
    text_features = text_features.reshape(N, C, 1)#([26, 512, 1])
    #print("########text_features:",text_features.size())
    

    similarity = torch.matmul(image_features, text_features)#torch.Size([26, 1, 1])
    #print("########similarity:",similarity.size())

    return similarity


def clip_forward(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)
    #print("########texts:",len(texts))
   # print("########texts:",texts)
    #print("#######images:",images.shape)
    _, _,H,W = images.shape
    texts = clip.tokenize(texts).cuda()#([26, 77])

    #print("########texts:",texts.size())

    image_features,_ = clip_model.encode_image(images,H,W)#([26, 512])#image_features: torch.Size([50, 27, 768])
    #image_features.requires_grad = True  # 允许梯度
    #print("########image_features:",image_features.size())
    #image_features, attn_weight = clip_model.transformer.resblocks(image_features)
    #print("########image_features:",image_features.size())
    x = image_features.permute(1, 0, 2)  # LND -> NLDx: torch.Size([27, 50, 768])
    #print("########xx:",x.size())
    #x = clip_model.visual.ln_post(x)
    #image_features = torch.mean(x[:,1:,:],dim=1)
    image_features = clip_model.visual.ln_post(x[:, 0, :])

    # ---- 关键：如果视觉侧有 proj，就把 768 -> 512 ----
    if getattr(clip_model.visual, "proj", None) is not None:
        # 一般 CLIP 模型都会有 self.visual.proj
        image_features = image_features @ clip_model.visual.proj


    #print("########image_features22:",image_features.size())
    text_features = clip_model.encode_text(texts)#([26, 512])
    #print("########text_features:",text_features.size())

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)#([26, 1, 512])
    #print("########image_features33:",image_features.size())
    #print("########image_features:",image_features.size())
    #print("########text_features:",text_features.size())
    text_features = text_features.reshape(N, C, 1)#([26, 512, 1])
    #print("########text_features:",text_features.size())
    

    similarity = torch.matmul(image_features, text_features)#torch.Size([26, 1, 1])
    #print("########similarity:",similarity.size())

    return similarity

"""class CLIPContrastiveLoss(torch.nn.Module):
    def __init__(self, clip_model, dataset='voc', temperature=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.dataset = dataset
        self.temperature = temperature
        #self.category_names = category_dict[dataset]"""

def clip_forward_F(clip_model, images, labels, dname='voc', temperature=0.07):
        # images: [N, 3, 224, 224]
        # labels: [N, 20] one-hot
        
        # Text prompt for all categories
        _, _,H,W = images.shape
        texts = to_text(labels, dname)
        #print("#########texts:",texts.shape)
        texts = clip.tokenize(texts).cuda()

        text_features = clip_model.encode_text(texts)

        # normalized features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        """all_class_names = [f'a photo of a {cls}' for cls in self.category_names]
        class_texts = clip.tokenize(all_class_names).cuda()  # [20, 77]
        with torch.no_grad():
            text_features = self.clip_model.encode_text(class_texts)  # [20, D]
            text_features = F.normalize(text_features, dim=1)"""

        # Encode images
        image_features,_ = clip_model.encode_image(images,H,W)
        x = image_features.permute(1, 0, 2)  # LND -> NLDx: torch.Size([27, 50, 768])
        image_features = clip_model.visual.ln_post(x[:, 0, :])

        # ---- 关键：如果视觉侧有 proj，就把 768 -> 512 ----
        if getattr(clip_model.visual, "proj", None) is not None:
            # 一般 CLIP 模型都会有 self.visual.proj
            image_features = image_features @ clip_model.visual.proj
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        #image_features = self.clip_model.encode_image(images)  # [N, D]
        #image_features = F.normalize(image_features, dim=1)

        # Compute similarities: [N, 20]

        N, C = image_features.size()
        #image_features = image_features.reshape(N, 1, C)
        #text_features = text_features.reshape(N, C, 1)
        
        logits = image_features @ text_features.T  # cosine similarity
        logits = (logits / temperature).view(N,-1)
        #print("#########logist:",logits.shape)

        # Create target: for each row i, positive label index where labels[i, c] == 1
        #print("#########labels:",labels.shape)
        targets = labels.argmax(dim=1)  # [N] → index of positive class
        #print("#########targets:",targets.shape)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        return loss

def clip_forward3(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)

def clip_forward2(clip_model, images, labels, dname='coco'):
    
    text_features = labels
    image_features = clip_model.encode_image(images).detach()#([20, 512])
    #print("########image_features:",image_features.size())

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)#([20, 1, 512])
    #print("########image_features:",image_features.size())
    text_features = text_features.reshape(N, C, 1)#([20, 512, 1])
    #print("########text_features:",text_features.size())
    

    similarity = torch.matmul(image_features, text_features)#torch.Size([20, 1, 1])
    #print("########similarity:",similarity.size())

    return similarity