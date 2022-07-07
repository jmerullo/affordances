from torch.utils import data
import torch
torch.manual_seed(0)
import json
import os
import numpy as np
import random
from torchvision.transforms import Normalize

import sys
sys.path.append('..')
from constants import aff_order_by_name, OBJ_AFFS
from rich.progress import track
from dataclasses import dataclass

@dataclass
class Sequence:
    seq_id: str
    corners: np.array
    objname: str
    force:   np.array
    mass:    float
    drag:    float
    ang_drag:float


def rot_matrix(deg): #rotation around the y-axis. the angle of the force is around x-z
    rads = np.radians(deg) #start with degrees because it's easier to think about
    return torch.tensor([[np.cos(rads), 0, -np.sin(rads)], [0, 1, 0], [np.sin(rads), 0, np.cos(rads)]]).float()


class SeqLoader(data.DataLoader):

    def __init__(self, data_folder, object_names, rotnoise=False, coords_only=True, 
            mask_rate=0.0, skip=1, src_stop_point=-1,dummy=False, dont_load=False, hard_obj_limit=10000,
            apply_src_pad_mask=False, pred_len=1):
        #data_folder: str that contains all of the object folders with sequence json files
        #super(SequenceLoader, self).__init__()

        coords_inpt_size=27 #TODO change from 27
        self.coords_inpt_size = coords_inpt_size

        self.PAD = torch.zeros(coords_inpt_size)
        self.MASK= torch.ones(coords_inpt_size)*-1
        self.MIN_SEQ_LEN = 30*skip
        self.SEQ_LEN = 240
        self.seqdata = []
        self.pathdata =[]
        self.skip=skip
        self.src_stop_point=src_stop_point
        self.mask_rate=mask_rate
        self.apply_src_pad_mask=apply_src_pad_mask
        self.pred_len=pred_len #how many frames to predict
        self.prof_path = "multiview/"
        self.multiviews = {}
        #training setup bools
        self.coords_only=coords_only
        if not coords_only:
            self.num_add_inputs = 6
        else:
            self.num_add_inputs = 0
        self.dummy = dummy
        self.rotnoise=rotnoise
        if rotnoise:
            print("PREPROCESSING WITH ROTNOISE (rotations around y)") 
        self.objnames = object_names#[:1]

        if dont_load:
            return

        print('objnames', self.objnames)
        for objfile in os.listdir(data_folder):
            objname = objfile[:-4]
            if objname not in self.objnames or not objfile.endswith('npz'):
                print("CONTINUING", objname)
                continue
            seqpath = os.path.join(data_folder, objfile)
            npdata = np.load(seqpath, allow_pickle=True)
            seqs = npdata['seqs']

            skip_cntr=0
            print('obj len', objname, len(seqs))
            if hard_obj_limit != -1:
                seqs = seqs[:hard_obj_limit]
            for seq in track(seqs, description='loading {}'.format(objname), style='blue'):
                #frames = data[seqcode]
                corners = self.parse_corners(seq['frames'])
                if len(corners)<min(self.MIN_SEQ_LEN, 240):
                    skip_cntr+=1
                    continue

                seq_id  = seq['seq_id']
                force   = seq['force']
                mass    = seq['mass']
                drag    = seq['drag']
                ang_drag= seq['angular_drag']
                if not self.coords_only:
                    self.multiviews[objname]=np.load(os.path.join(self.prof_path, objname+'.npy'))

                #Append a sequence dataclass object
                self.seqdata.append(Sequence(seq_id, corners, objname, force, mass, drag, ang_drag))
            print(skip_cntr, "SKIPPED")
        print("TOTAL DATA", len(self.seqdata))

    def parse_corners(self, frames):
        frame_corners=[]
        for f in frames:
            corners = f['corners']
            '''
            corn=[]
            corners = f['corners']
            for c in corners:
                corn.append([c['x'], c['y'], c['z']])
            '''
            frame_corners.append(corners)
        return np.stack(frame_corners)


    def affs_from_obj(self, objname):
        try:
            return np.array(OBJ_AFFS[objname])
        except KeyError as e:
            print(e)
            return np.array([int('obj_0' in objname.lower() or 'obj_1' in objname.lower()),int(not('obj_0' in objname.lower() or 'obj_1' in objname.lower())),0,0,0,0])


    def get_profs(self, profdir):
        profs = []#the profs are already encoded in some small form
        for fname in os.listdir(profdir):
            fpath = os.path.join(profdir, fname)
            profs.append(np.load(fpath))
        return profs

    def __len__(self):
        return len(self.seqdata)

    def __getitem__(self, index, debug=False):
        #need coords sequence (masked and unmasked), object profiles, and physics profile, objname
        #need to pad coords sequence, and return attn mask

        #coords data
        seqlen = self.SEQ_LEN
        seq    = self.seqdata[index]
        seq_id, corners, objname, force, mass, drag, ang_drag = seq.seq_id, seq.corners, seq.objname, seq.force, seq.mass, seq.drag, seq.ang_drag
        affs = self.affs_from_obj(objname)

        centers = corners.mean(axis=1) #n_frames,3
        coords  = np.insert(corners, corners.shape[1], centers, axis=1)
        assert coords.shape[1] == 9 and coords.shape[2]==3
        seq = torch.tensor(coords.reshape(-1, 27))
        if len(seq)>seqlen:
            seq = seq[:seqlen]
        if self.rotnoise:
            seq = seq.view((-1,3))
            angle = random.randint(0,360)
            first_point=seq[0].clone()
            translation_mat=torch.zeros_like(seq)-first_point
            inv_translation_mat = torch.zeros_like(seq)+first_point
            rot = rot_matrix(angle) #rotation around the y axis, start with degrees
            seq_translated = torch.mm(torch.add(translation_mat, seq), rot) #rotate around the first point in the sequence
            seq = torch.add(inv_translation_mat, seq_translated)
            seq = seq.view((-1, self.coords_inpt_size))

        if len(seq)>self.MIN_SEQ_LEN:
            seq = seq[::self.skip]
        tgt = seq[-self.pred_len:].clone()
        src = seq[:-self.pred_len].clone()
        if len(src)>self.MIN_SEQ_LEN:
            src = src[::self.skip]

        if self.mask_rate>0:
            n_masks = max(1, int((int(len(src)-1)*self.mask_rate)))
            mask_start = random.randint(0, (len(src)-n_masks)-1)
            mask_end=mask_start+n_masks
            src[mask_start:mask_end]=self.MASK

            
        padding = seqlen-(len(src))

        if self.apply_src_pad_mask:
            padding = [self.PAD]*padding
        else:
            padding = [src[-1]]*padding
        if len(padding) == 0:
            padding = torch.zeros(0, self.PAD.shape[0])
        else:
            padding = torch.stack(padding)

        src = torch.cat([src, padding], dim=0)


        if len(tgt.shape)==1:
            tgt = torch.stack((self.MASK, tgt))
        else:
            tgt = torch.cat((self.MASK.unsqueeze(0), tgt), dim=0)

        physics = np.concatenate([force, [mass]])
        physics = torch.tensor(physics)

        #profile data
        profs=[0]
        prof_label = 1 #True - the object matches the sequence
        if not self.coords_only:
            OBJS = list(self.multiviews.keys())
            if random.random() > .5: #TODO CHANGE THIS
                OBJS.remove(objname)
                my_roll_aff = OBJ_AFFS[objname][1]
                OBJS = [o for o in OBJS if OBJ_AFFS[o][1]!=my_roll_aff]
                chosen_obj = random.choice(OBJS)
                profs = self.multiviews[chosen_obj] #Adversarial case
                prof_label=0 #False
            else: 
                #object matches the sequence case
                profs = self.multiviews[objname]#torch.tensor([0]) #TODO: Make this

        if self.apply_src_pad_mask:
            src_pad_mask = torch.ones(len(src)+self.num_add_inputs).float()#torch.ones(len(seq)+len(profs)+1).float()
            src_pad_mask[:len(src)-len(padding)]=0.
            #src_pad_mask[len(vanilla):]=0.0
        else:
            src_pad_mask = torch.zeros(len(src)+self.num_add_inputs)
        src_pad_mask = src_pad_mask.bool()


        objname = objname
        if debug:  
            return seq, src, tgt, torch.tensor(profs), physics, src_pad_mask, objname, self.pathdata[index], self.profpaths[index], prof_label
        else:
            return src, tgt, torch.tensor(profs), physics, src_pad_mask, objname, seq_id, affs, prof_label


