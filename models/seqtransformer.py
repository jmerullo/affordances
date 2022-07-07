import math
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet34
OBJ_AFFS={
    "BombBall":             [0,1,0,0,0,1],
    "EyeBall":              [0,1,0,0,0,1],
    "SpikeBall":            [0,1,0,0,0,0],
    "Vase_Amphora":         [0,1,0,0,0,0],
    "Vase_Hydria":          [0,1,0,0,0,0],
    "Vase_VoluteKrater":    [0,1,0,1,0,0],
    "book_0001a":           [1,0,1,0,0,0],
    "book_0001b":           [1,0,1,0,0,0],
    "book_0001c":           [1,0,1,0,0,0],
    "bowl01":               [1,1,1,1,0,0],
    "cardboardBox_01":      [1,0,1,0,0,0],
    "cardboardBox_02":      [1,0,1,1,0,0],
    "cardboardBox_03":      [1,0,1,0,0,0],
    "Cola Can":             [1,1,1,0,1,0],
    "Pen black":            [0,1,0,0,1,0],
    "Gas Bottle":           [0,1,0,0,0,0],
    "Soccer Ball":          [0,1,0,0,0,1],
    "can small":            [1,1,1,0,1,0],
    "can":                  [1,1,1,0,1,0],
    "meat can box":         [1,0,1,0,0,0],
    "spam can":             [1,0,1,0,1,0],
    "AtomBall":             [0,1,0,0,0,1],
    "Bottle2":              [0,1,0,0,1,0],
    "plate02":              [1,0,1,0,0,0],
    "plate02_flat":         [1,0,1,0,0,0],
    "Bottle1":              [0,1,0,0,1,0],
    "WheelBall":            [0,1,0,0,0,1],
    "wine bottle 04":       [0,1,0,1,1,0],
    "coin":                 [1,0,1,0,0,0],
    "BuckyBall":            [0,1,0,0,0,1],
    "SplitMetalBall":       [0,1,0,0,0,1],
    "bowl02":               [1,1,1,1,0,0],
    "bowl03":               [1,1,1,1,0,0],
    "mug02":                [1,0,0,1,1,0],
    "mug03":                [1,0,0,1,1,0],
    "Old_USSR_Lamp_01":     [1,0,0,0,1,0],
    "lamp":                 [1,1,0,0,1,0],
    "Ladle":                [1,0,0,0,1,0],
    "Apple":                [0,1,0,0,0,0],
}


class CNNImgEncoder(nn.Module):

    def __init__(self, emb_size):
        super(CNNImgEncoder, self).__init__()
        net = resnet34(pretrained=True)
        self.conv1=net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool= net.maxpool
        self.layer1 = net.layer1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, emb_size)
        self.emb_size = emb_size


    def forward(self, x):
        #print("CNN INPUT", x.shape)
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.view(-1, 6, self.emb_size).transpose(0,1)


#Use this if loading the image encodings from file, in the case resnet already encoded them
class LinearImgEncoder(nn.Module):
    def __init__(self, emb_size):
        super(LinearImgEncoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(512, emb_size), nn.ReLU())
        #nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, emb_size))
        self.bn =  nn.BatchNorm1d(emb_size)
        self.input_size = 512
    def forward(self, x):
        
        x= self.net(x)
        #x=torch.transpose(x, 1,2)
        #x = self.bn(x)
        #x=torch.transpose(x, 1,2)
        return torch.transpose(x, 0,1)

class SeqTransformer(nn.Module):

    def __init__(self, config, num_heads=1):
        super(SeqTransformer, self).__init__()
        #TODO change
        self.input_size = 27
        self.tgt_len=config['pred_len'] #start token and then the first frame. Was 1
        additional_inputs=2 #just the cls and sep tokens
        if config['include_multiview']:
            additional_inputs+=6 # sep and the 6 image frames
        self.include_multiview = config['include_multiview']
        self.max_seq_len = config['max_seq_len']+additional_inputs #TODO: uncomment the plus two #add two for the CLS and SEP tokens
        self.batch_size  = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.input_encoder = nn.Linear(self.input_size, self.emb_dim)
        #TODO uncomment
        self.embeddings = nn.Embedding(2, self.emb_dim)
        #self.bn = nn.BatchNorm1d(self.emb_dim)
        self.prof_encoder = LinearImgEncoder(self.emb_dim)#CNNImgEncoder(self.emb_dim)#
        self.posenc = PositionalEncoding(self.emb_dim, config['dropout'], max_len=self.max_seq_len)
        self.transformer = nn.Transformer(d_model=self.emb_dim, dim_feedforward=config['dim_feedforward'], 
            num_encoder_layers=config['num_encoder_layers'], num_decoder_layers=config['num_decoder_layers'],
            nhead=config['num_heads'], batch_first=False, dropout=config['dropout'])

        self.decoder = nn.Linear(self.emb_dim, self.input_size)
        self.cls_network = nn.Sequential(nn.Linear(self.emb_dim,50), nn.ReLU(), nn.Linear(50, 2))
        #self.physics_input_encoder=nn.Linear(4, emb_dim)



    def add_cls_sep_tokens(self, x):
        cls_i = torch.zeros(x.shape[1]).long().cuda()
        sep_i = torch.ones(x.shape[1]).long().cuda()
        x[0]=self.embeddings(cls_i)
        x[-7] = self.embeddings(sep_i)
        return x

    def encode_only_imgs(self, profs):
        profs = self.prof_encoder(profs)
        sep_i = torch.ones(1).long().cuda()
        sep = self.embeddings(sep_i)
        x = torch.cat([sep, profs], dim=0)
        encs = self.transformer.encoder(x)
        return encs

    def forward(self, src, tgt, src_key_padding_mask, profs, prof_labels):
        if self.include_multiview:
            return self.forward_with_mv(src, tgt, src_key_padding_mask, profs, prof_labels)
        else:
            return self.forward_coords_only(src, tgt, src_key_padding_mask)
    
    def encoding(self, src, src_key_padding_mask ,profs):
        src = self.posenc(self.add_cls_sep_tokens(self.input_encoder(src)))

        profs = self.prof_encoder(profs)
        src = torch.cat((src,profs), dim=0) #sequence first
        e = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return e

    def mv_encoding(self, src, tgt, src_key_padding_mask ,profs):
        enc = self.encoding(src, src_key_padding_mask ,profs)
        enc = enc[-6:]
        return enc

    def forward_with_mv(self, src, tgt, src_key_padding_mask, profs, prof_label):
        src = self.posenc(self.add_cls_sep_tokens(self.input_encoder(src)))
        tgt = self.posenc(self.input_encoder(tgt))

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).cuda()

        profs = self.prof_encoder(profs)

        src = torch.cat((src,profs), dim=0)
        
        e = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        t = self.transformer.decoder(tgt, e, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        t = torch.transpose(t, 0, 1)

        
        cls_embs = e[0] #CLS encoding

        clf_logits = self.cls_network(cls_embs)

        return self.decoder(t), clf_logits, e


    def forward_coords_only(self, src, tgt, src_key_padding_mask):
        #since input size is 2, no need for a tgt_key_padding mask. Also no need for a src_mask rn
        #src needs to be (s,n,e) tgt needs to be shape (t,n,e) (27, batch, e) and (2, batch, e) respectively

        

        src = self.posenc(self.add_cls_sep_tokens(self.input_encoder(src))) #(N, 128) -> (N, 128, 300) or transposed depending on batch_first
        tgt = self.posenc(self.input_encoder(tgt)) #(N, 1)
        


        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).cuda()
        
        t = self.transformer(src, tgt , tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        t = torch.transpose(t, 0, 1)
        #TODO: double check view is doing what i want it to
        #t = t.view(-1, self.tgt_len, self.emb_dim)
        encs = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(t), None, encs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    def forward_nodropout(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

