import torch

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            'dog',
            'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'],
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
#prompt_dict = ['a clean origami {}.']


def to_text(labels, dataset='voc'):
    _d = category_dict[dataset]

    text = []
    #########labels.size:([22, 20])
    print("########labels.size(0):",labels.size(0))
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    
    text_encoder = text_encoder.cuda()
    # text_encoder = text_encoder.cuda()
    
    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in prompt_dict])
            tokens = tokens.to(device)
            # print("=============", tokens.dtype, clip_model.dtype)
              # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


import clip
def clip_forward(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)
    #print("########texts:",len(texts))
   # print("########texts:",texts)
    texts = clip.tokenize(texts).cuda()#([26, 77])

    #print("########texts:",texts.size())

    image_features = clip_model.encode_image(images)#([26, 512])
    #print("########image_features:",image_features.size())
    text_features = clip_model.encode_text(texts)#([26, 512])
    #print("########text_features:",text_features.size())

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)#([26, 1, 512])
    #print("########image_features:",image_features.size())
    text_features = text_features.reshape(N, C, 1)#([26, 1, 512])
    #print("########text_features:",text_features.size())
    

    similarity = torch.matmul(image_features, text_features)

    return similarity
