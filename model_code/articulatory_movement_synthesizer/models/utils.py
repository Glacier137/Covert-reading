import os
from tkinter.tix import Tree
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import dill
import torch.utils.data as data
from os import read
from numpy import dtype
import torch.utils.data as data
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import dill

# --------------------------------------------- #
#                  Data utilities
# --------------------------------------------- #


def load_data(args):
    
    if args.is_infer==True:
        with open(os.path.join(args.dataset_path,f"HS{args.HS}_{args.reading_name}_train_loader_{args.band}_{args.elec_type}_infer_{args.used_sound}.pkl"),'rb') as f:
            train_loader = dill.load(f)
        with open(os.path.join(args.dataset_path,f"HS{args.HS}_{args.reading_name}_val_loader_{args.band}_{args.elec_type}_infer_{args.used_sound}.pkl"),'rb') as f:
            val_loader= dill.load(f)
        test_loader= []
    elif args.is_percentage==True:
        band = args.band if hasattr(args, 'band') else 'high_gamma'
        train_filename = f"HS{args.HS}_{args.reading_name}_train_loader_{band}_{args.elec_type}_fold{args.fold_ind}_percentage_{args.percent}.pkl"
        val_filename = f"HS{args.HS}_{args.reading_name}_val_loader_{band}_{args.elec_type}_fold{args.fold_ind}_percentage_{args.percent}.pkl"
        with open(os.path.join(args.dataset_path, train_filename),'rb') as f:
            train_loader = dill.load(f)
        with open(os.path.join(args.dataset_path, val_filename),'rb') as f:
            val_loader= dill.load(f)
        test_loader= []
    elif args.is_fold==True:
        band = args.band if hasattr(args, 'band') else 'high_gamma'
        train_filename = f"HS{args.HS}_{args.reading_name}_train_loader_{band}_{args.elec_type}_fold{args.fold_ind}.pkl"
        val_filename = f"HS{args.HS}_{args.reading_name}_val_loader_{band}_{args.elec_type}_fold{args.fold_ind}.pkl"
        with open(os.path.join(args.dataset_path, train_filename),'rb') as f:
            train_loader = dill.load(f)
        with open(os.path.join(args.dataset_path, val_filename),'rb') as f:
            val_loader= dill.load(f)
        test_loader= []
        
    else:
        with open(os.path.join(args.dataset_path,f"HS{args.HS}_{args.reading_name}_train_loader_beta_{args.elec_type}.pkl"),'rb') as f:
            train_loader = dill.load(f)
        with open(os.path.join(args.dataset_path,f"HS{args.HS}_{args.reading_name}_val_loader_beta_{args.elec_type}.pkl"),'rb') as f:
            val_loader= dill.load(f)
        with open(os.path.join(args.dataset_path,f"HS{args.HS}_{args.reading_name}_test_loader_beta_{args.elec_type}.pkl"),'rb') as f:
            test_loader= dill.load(f)

    return train_loader,val_loader,test_loader


# --------------------------------------------- #
#                 Module utilities
#              for encoders and decoders
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
