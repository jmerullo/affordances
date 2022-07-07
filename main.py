import argparse
import os

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils import data

#import dataloaders
from dataloaders.seqloader import SeqLoader

#import Models
from models.seqtransformer import SeqTransformer


#import rich settings
from rich.console import Console
from rich.json import JSON
from rich.traceback import install
install(show_locals=False)


import json
import wandb
from datetime import datetime
from utils import get_device, save_checkpoint, load_checkpoint, check_gpu_mem, aff_labels_from_objnames
import model_wrapper
from model_wrapper import PretrainWrapper, Results, PredLabels
from constants import OBJ_AFFS, aff_order_by_name


console=Console(record=True, force_terminal=True)

def create_dl(config, is_val, shuffle=True): 
    batch_size, coords_only, is_dummy, gnoise, rotnoise, mask_rate, skip, src_stop_point = config['batch_size'], config['coords_only'], config['is_dummy'], config['gnoise'],config['rotnoise'], config['mask_rate'], config['skip'], config['src_stop_point']
    
    
    if is_val:
        datadir = config['dataset']['val_path']
        objs = config['dataset']['val_objs']
        console.print('creating dl with configuration:', style='bold')
        #console.print('Dataset', JSON(json.dumps(config['dataset'])), style='bold red')
        console.print("batch size", batch_size)
        console.print("is val?", is_val)
        console.print('coords only?', coords_only)
        console.print('is dummy?', is_dummy)
        console.print('translation noise?', gnoise)
        console.print('rotation noise?', rotnoise)
        console.print('mask rate:', mask_rate)
        console.print('skip', skip)
        console.print('src_stop_point', src_stop_point)
        console.print('shuffle?', shuffle)
        console.print("hard obj limit", config['hard_obj_limit'])
        console.print("Apply src pad mask?", config['apply_src_pad_mask'])
        console.print("Pred len", config['pred_len'])
    else:
        datadir = config['dataset']['train_path']
        objs = config['dataset']['train_objs']
    dl = SeqLoader(datadir, objs, coords_only=coords_only, mask_rate=mask_rate, 
            skip=skip, src_stop_point=src_stop_point, rotnoise=rotnoise, 
            hard_obj_limit=int(config['hard_obj_limit']), 
            apply_src_pad_mask=config['apply_src_pad_mask'], pred_len=config['pred_len'])
    
    loader = data.DataLoader(dataset=dl, 
                    batch_size=batch_size, 
                    shuffle=shuffle) 
    console.print("Length of dataloader:", len(loader))
    return loader



def init_training(run_config):

    #initialize import variables, not necessarily used but required in run
    console.rule("INIT TRAINING", style='yellow')
    device=get_device()
    net = SeqTransformer(run_config)
        
    if 'load_model' in run_config:
        console.log("Loading Model - [red]{}".format(run_config['modelname']))
        net = load_checkpoint(net, run_config['modelname']+'.pth')
    lr = run_config['lr']

    #define optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    #Create dataloaders
    val_dl   = create_dl(run_config, is_val=True)
    train_dl = create_dl(run_config, is_val=False)
    
    if run_config['wandb']:
        wandb.watch(net, log='all')

    wrapper = PretrainWrapper()
    wrapper.net = net.to(device)
    wrapper.optimizer=optimizer
    wrapper.loss_fn = loss_fn
    wrapper.train_dl= train_dl
    wrapper.val_dl  = val_dl

    wrapper.run_config=run_config
    return wrapper

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default='configs/run_default.json')
    parser.add_argument('--pipeline_config', type=str, default='configs/pipe_default.json')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--is_sweep", action='store_true')
    parser.add_argument('--steps', type=int, default=25)
    #sweep params
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--dim_feedforward", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask_rate", type=float, default=0.)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--src_stop_point", type=int, default=-1)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--include_multiview', type=int, default=0) #default false
    parser.add_argument("--force_seq", type=int, default=0)
    return parser

def read_json(path):
    fp = open(path, 'r')
    data=json.load(fp)
    fp.close()
    return data

def save_json(data, path):
    fp = open(path, 'w')
    json.dump(data, fp, indent=4)
    fp.close()

def main():
    global console
    parser = create_argparser()
    args = parser.parse_args()
    console=Console(record=True, force_terminal=True)
    device=get_device()

    #get the run config, holds all hyperparams, paths, etc.
    run = read_json(args.run_config)
    run['debug'] = args.debug 
    run['wandb'] = args.wandb
    run['dataset']=read_json(run['dataset'])
    if args.is_sweep:
        run['is_sweep']        = args.is_sweep
        run['batch_size']      = args.batch_size
        run['dim_feedforward'] = args.dim_feedforward
        run['dropout']         = args.dropout
        run['lr']              = args.lr
        run['mask_rate']       = args.mask_rate
        run['num_encoder_layers']=args.num_layers
        run['num_decoder_layers']=args.num_layers
        run['skip']            = args.skip
        run['src_stop_point']  = args.src_stop_point
        run['pred_len']        = args.pred_len
        run['include_multiview']=bool(int(args.include_multiview))
        run['steps']           = args.steps
        run['force_seq']       = bool(int(args.force_seq))
    run['coords_only']=not run['include_multiview']
    #get the driver config (what's actually going to be run)

    if run['wandb']:
        wandb_run = wandb.init(
            project='unity transformer',
            config=run)
        #run = wandb_run.config
    console.rule(f"Run at: {datetime.now().ctime()}")
    console.print("RUN CONFIG")
    console.print(run)

    best_loss = 1000000
    best_encs = []
    best_encs_aff_labels = []
    best_objnames = []
    best_step = 0
    best_vmeas= 0
    best_vresults=None
    best_tresults=None
    modelname = run['modelname']

    mw = init_training(run)#PretrainWrapper(init_training(run))

    #get the number of steps to repeat the pipeline
    steps = run['steps']

    torch.cuda.empty_cache()
    for cur_step in range(steps):
        train_r, val_r = mw.traineval_step()
        log_metrics = {"trainloss":train_r.loss, "valloss":val_r.loss}
        if not run['is_sweep']:
            console.print(log_metrics)
        if val_r.loss<best_loss:
            best_tresults = train_r
            best_vresults  = val_r
            best_loss = val_r.loss
            best_step =cur_step
            best_objnames = val_r.objnames 
            if not run['is_sweep']:
                state = {'model':mw.net.state_dict(), 'valloss':val_r.loss, 'epoch':cur_step}
                save_checkpoint(state, modelname+".pth")
        
        if run['wandb']:
            wandb.log(log_metrics)

    console.log("After loop...")

    if run['wandb']:
        wandb_run.finish()
    

if __name__=='__main__':
    main()
