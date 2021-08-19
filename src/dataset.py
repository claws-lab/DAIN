#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2021-08-17
#Influence-guided Data Augmentation for Neural Tensor Completion (DAIN)
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import numpy as np
import pandas as pd
import torch
import collections
from torch.utils.data import DataLoader,Dataset

class TensorDataset(Dataset):
    def __init__(self, file_name):
        X = pd.read_csv(file_name,sep='\t',header=None).values
        self.tensor,self.val,self.indices = torch.LongTensor(X[:,:-1]),torch.FloatTensor(X[:,-1]),torch.zeros(X.shape[0]).long()
        self.maxx,self.minn = torch.max(self.val),torch.min(self.val)
        self.mean,self.std = torch.mean(self.val).item(),torch.std(self.val).item()
        self.num_data, self.order = self.tensor.shape[0],self.tensor.shape[1]
        self.dimensionality = [int(max(self.tensor[:,i]))+1 for i in range(self.order)]
    
    def add(self,new_tensor,new_val,new_indices):
        self.tensor = torch.cat((self.tensor,new_tensor),0)
        self.val = torch.cat((self.val,new_val),0)
        self.indices = torch.cat((self.indices,new_indices),0)
        self.num_data, self.order = self.tensor.shape[0],self.tensor.shape[1]

    def delete(self, indices):
        self.tensor = torch.from_numpy(np.delete(self.tensor.numpy(),indices,axis=0))
        self.val = torch.from_numpy(np.delete(self.val.numpy(),indices,axis=0))
        self.num_data = self.tensor.shape[0]

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        return self.tensor[index,:],self.val[index],self.indices[index]

