#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2021-08-17
#Influence-guided Data Augmentation for Neural Tensor Completion (DAIN)
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.


import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, dataset, device,layers):
        super().__init__()

        assert (layers[0] % dataset.order == 0), "layers[0] (=order*embedding_dim) must be divided by the tensor order"
        self.device = device

        embedding_dim = int(layers[0]/dataset.order)

        self.embeddings = nn.ModuleList()
        for i in range(dataset.order):
            self.embeddings.append(torch.nn.Embedding(dataset.dimensionality[i],embedding_dim))
        
        # list of weight matrices
        self.fc_layers = nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.relu = nn.ReLU() 
        # final prediction layer
        self.last_size = layers[-1]
        self.output_layer = nn.Linear(layers[-1], 1)

    def forward(self, x):
        embeddings = [self.embeddings[i](x[:,i]) for i in range(len(self.embeddings))]
        # concatenate embeddings to form input
        x = torch.cat(embeddings, 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = self.relu(x)
        self.intermediate = x
        out = self.output_layer(x)
        del embeddings
        return out

    def predict(self, tensor,batch_size=1024):
        k = int(tensor.shape[0]/batch_size)
        final_output = torch.zeros(tensor.shape[0])
        for i in range(k+1):
            st_idx,ed_idx = i*batch_size,(i+1)*batch_size
            if ed_idx>tensor.shape[0]:
                ed_idx = tensor.shape[0]
            if st_idx>=ed_idx:
                break
            idx = torch.LongTensor(list(range(st_idx,ed_idx)))
            x = tensor[idx,:].clone().to(self.device)
            final_output[idx] = self(x).flatten().cpu().detach().clone()
            del x,self.intermediate
            torch.cuda.empty_cache() 

        return final_output

