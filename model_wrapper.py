from utils import get_device, check_gpu_mem, save_checkpoint
import torch
from dataclasses import dataclass
import numpy as np
from rich.progress import track
import sys
from torch.utils import data
sys.path.append('..')

from sklearn.metrics import precision_recall_fscore_support

from collections import Counter
import random

@dataclass
class PredLabels:
    preds: np.array
    labels: np.array
    

    def precision_recall_f1_support(self, avg='binary'):
        return precision_recall_fscore_support(self.labels, self.preds, beta=1.0, average=avg,zero_division=0) 

    def convert_logits(self):
        self.preds = self.preds.argmax(axis=1)
"""
@dataclass
class Results:
    trainloss: float
    valloss: float
    trainpredlabels: PredLabels
    valpredlabels: PredLabels
    training_encs: np.array
    val_encs: np.array
    aff_labels: np.array
    train_objnames: list
    val_objnames: list
    comp: float=0.0
    homo: float=0.0
    vmeas: float=0.0
    baseline: float=0.0
"""
@dataclass
class Results:
    loss: float
    predlabels: PredLabels
    encs: np.array
    aff_labels: np.array
    objnames: list
    seq_ids: list
    comp: float=0.0
    homo: float=0.0
    vmeas: float=0.0
    baseline: float=0.0

@dataclass
class Encoding:
    enc: np.array
    objname: str
    seq_id: str


class PretrainWrapper:

    def __init__(self):
        self.net = None
        self.optimizer=None
        self.loss_fn = None
        self.train_dl = None
        self.val_dl = None
        self.run_config=None
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.last_seqids = None
        self.clf_loss_fn = torch.nn.CrossEntropyLoss()

        self.training_encs=[]
        self.val_encs= []
        self.clf_predlist = []
        self.clf_labellist= []


    def run_model(self, src, tgt, src_pad_mask, profs, prof_labels, objs, is_val=False):
        #clf_logits is None in the case where multiview is not included
        preds, clf_logits, encs = self.net(src, tgt[:-1, :], src_pad_mask, profs, prof_labels) #tgt = [pred_len, N] sample tgt: [MASK, 12] ... was: tgt[:-1, :]

        labels=torch.transpose(tgt[1:,:], 0, 1)

        
        if clf_logits is not None:
            clf_loss = self.clf_loss_fn(clf_logits, prof_labels)
            self.clf_predlist.append(clf_logits.detach().cpu().numpy())
            self.clf_labellist.append(prof_labels.detach().cpu().numpy())

        pred_loss = self.loss_fn(preds, labels)
        if clf_logits is not None:
            loss = pred_loss+(clf_loss)
        else:
            loss = pred_loss
        
        if self.run_config['include_multiview'] and not self.run_config['force_seq']:
            encs = encs[-6:]#self.net.mv_encoding(src, tgt, src_pad_mask, profs)
        elif self.run_config['coords_only']:
            print("COORDS ONLY")
            print(encs.shape)
            encs = encs#self.net.encoding(src, src_pad_mask, profs)[:-6] #exclude img profs
        else:
            encs = encs[:-6]

        #expected (S, N, E) = (130, 1, 100)
        #this is the avg of all in the sequence
        encs = encs.mean(dim=0)
        if is_val:
            self.val_encs.append(encs.detach().cpu().numpy())
        else:
            self.training_encs.append(encs.detach().cpu().numpy())

        del src
        del clf_logits
        if self.run_config['include_multiview']:
            return preds, loss, clf_loss.item()
        return preds, loss, 0.0


    def fitpredict(self, dl, is_val):
        if is_val:
            self.net.eval()
        else:
            self.net.train()
        device = get_device()
        running_loss = 0.0
        running_clf_loss = 0.0
        dummy=self.run_config['is_dummy']
        i = 0
        saved_preds = []
        aff_labels =[]
        predlist = []
        labellist = []
        objnames = []
        seq_ids = []
        self.clf_predlist = []
        self.clf_labellist= []

        for src, tgt, profs, physics, src_pad_mask, objs, seq_id, affs, prof_labels in track(dl, description='Running model...'):
            i+=1
            if not is_val:
                self.optimizer.zero_grad()

            src, tgt, profs = src.to(device), tgt.to(device), profs.to(device)
            prof_labels = prof_labels.to(device)
            src_pad_mask = src_pad_mask.to(device)
            src = torch.transpose(src, 0, 1) 
            tgt = torch.transpose(tgt, 0, 1) #should be [seqlen, batch_size] #128, N and 2,N

            preds, loss, clf_loss = self.run_model(src, tgt, src_pad_mask, profs, prof_labels, objs, is_val=is_val)
            if not is_val:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_clf_loss+=clf_loss
            labellist.append(tgt[1:, :].detach().cpu().numpy())
            predlist.append(preds.detach().cpu().argmax(dim=1).numpy())
            aff_labels.append(affs.detach().cpu().numpy())
            for sid, name in zip(seq_id, objs):
                objnames.append(name)
                seq_ids.append(str(sid))
            
            del preds
            del src
            del tgt
            del loss
            del profs
            del prof_labels
            del src_pad_mask

            torch.cuda.empty_cache()

        
        if self.run_config['include_multiview']:
            self.clf_pl = PredLabels(np.concatenate(self.clf_predlist), np.concatenate(self.clf_labellist))   
            self.clf_pl.convert_logits()

        torch.cuda.empty_cache()
        labels = np.concatenate(labellist, axis=1)
        avgloss = running_loss/float(len(dl))
        aff_labels=np.concatenate(aff_labels)
        self.last_seqids = seq_ids
        return np.concatenate(predlist), labels, avgloss, aff_labels, objnames

    def traineval_step(self):
        self.training_encs = []
        self.val_encs = []
        trainpreds, trainlabels, trainlossavg, train_aff_labels, train_objnames = self.fitpredict(self.train_dl, False)
        train_seqids = self.last_seqids
        valpreds, vallabels, vallossavg, val_aff_labels, val_objnames = self.fitpredict(self.val_dl, True)
        val_seqids = self.last_seqids

        trainpredlabels = PredLabels(trainpreds, trainlabels)
        valpredlabels = PredLabels(valpreds, vallabels)
        self.val_encs = np.concatenate(self.val_encs, axis=0)
        self.training_encs = np.concatenate(self.training_encs, axis=0)

        train_r = Results(trainlossavg, trainpredlabels, self.training_encs, train_aff_labels, train_objnames, train_seqids)
        val_r = Results(vallossavg, valpredlabels, self.val_encs, val_aff_labels, val_objnames, val_seqids)
        return train_r, val_r

    def eval_step(self):
        self.val_encs = []
        valpreds, vallabels, vallossavg, val_aff_labels, val_objnames = self.fitpredict(self.val_dl, True)
        val_seqids = self.last_seqids
        valpredlabels = PredLabels(valpreds, vallabels)
        self.val_encs = np.concatenate(self.val_encs, axis=0)
        val_r = Results(vallossavg, valpredlabels, self.val_encs, val_aff_labels, val_objnames, val_seqids)
        return val_r


