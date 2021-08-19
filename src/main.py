#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2021-08-17
#Influence-guided Data Augmentation for Neural Tensor Completion (DAIN)
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
from dataset import TensorDataset
import torch.optim as optim
from model import MLP
import pandas as pd
import copy
import random
from sklearn.model_selection import train_test_split
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run DAIN for the MLP architecture")
    parser.add_argument('--path', nargs='?', default='data/synthetic_10K.tensor',
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[150,1024,1024,128]',
                        help="Size of each layer. Note that the first layer is the concatenation of tensor embeddings. So layers[0]/N (N=order) is the tensor embedding size.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Show performance per X iterations')
    parser.add_argument('--gpu', type=str, default='0',
                    help='GPU number')
    parser.add_argument('--output', type=str, default='demo.txt',
                    help = 'output name')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                    help = 'Ratio of training data')

    return parser.parse_args()

def model_train_and_test(args, model, train_loader, val_loader,test_loader,first):
    output_path = 'output/'+args.output
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    device = model.device

    min_val,min_test,min_epoch,final_model = 9999,9999,0,0
    for epoch in range(args.epochs):

        torch.cuda.empty_cache() 
        running_loss = 0.0
        train_loss,valid_loss = 0,0

        for i, data in enumerate(val_loader, 0):
            inputs, labels, indices = data[0].to(device), data[1].to(device),data[2]
            outputs = model(inputs).flatten()
            if first==True:
                inter = model.intermediate.cpu().detach().clone()
                error = (outputs - labels).reshape(-1,1).cpu().detach().clone()
                model.allgrad[epoch,indices,:] = torch.mul(inter,error)
            loss = criterion(outputs,labels)
            loss.backward()
            valid_loss += loss.item()
            del inputs,labels,outputs,model.intermediate
        valid_loss /= (i+1)

        test_loss, test_accuracy = 0,0
        for i, data in enumerate(test_loader, 0):
            inputs, labels,indices = data[0].to(device), data[1].to(device),data[2]
            prediction = model(inputs).flatten()
            loss = criterion(prediction,labels)
            loss.backward()
            test_accuracy += torch.sum(torch.pow((prediction-labels),2)).cpu().item()
            del inputs,labels,prediction,model.intermediate

        test_accuracy/=len(test_loader.dataset)    

        for i, data in enumerate(train_loader, 0):
            inputs, labels,indices = data[0].to(device), data[1].to(device),data[2]
            optimizer.zero_grad()
            outputs = model(inputs).flatten()
            if first==True:
                inter = model.intermediate.cpu().detach().clone()
                error = (outputs-labels).reshape(-1,1).cpu().detach().clone()
                model.allgrad[epoch,indices,:] = torch.mul(inter,error)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del inputs, labels, outputs,indices,model.intermediate
        train_loss /= (i+1)

        if epoch%args.verbose==0:
            print('[%d] Train loss: %.3f\tValid loss = %.6f\t(Test RMSE = %.6f)\t' % (epoch + 1, train_loss, valid_loss,test_accuracy))
            print('[%d] Train loss: %.3f\tValid loss = %.6f\t(Test RMSE = %.6f)\t' % (epoch + 1, train_loss, valid_loss,test_accuracy),file=open(output_path,"a"),flush=True) 
        
        if min_val<=valid_loss and epoch-min_epoch>=10:
            break

        if min_val>valid_loss:
            min_val = valid_loss
            min_test = test_accuracy
            min_epoch = epoch
            final_model = copy.deepcopy(model)
            final_model.allgrad = copy.deepcopy(model.allgrad)
            final_model.checkpoint = epoch+1
 
    print('Finished Training\nFinal Test RMSE = {} @ (Epoch,validation loss) ({},{})\n'.format(min_test,min_epoch,min_val))      
    print('Finished Training\nFinal Test RMSE = {} @ (Epoch,validation loss) ({},{})\n'.format(min_test,min_epoch,min_val), file=open(output_path, "a"),flush=True)
    del model
    return min_test,final_model

def data_augmentation(trainset,new_tensor,new_val,val_loader,test_loader,args,device):
    #Step 4: data augmentation
    if new_tensor.shape[0]!=0:
        cur_trainset = copy.deepcopy(trainset)
        new_indices = torch.zeros(new_tensor.shape[0]).long()
        cur_trainset.add(new_tensor,new_val,new_indices)
        first = False
    #Step 1: tensor embedding learning
    else:
        cur_trainset = copy.deepcopy(trainset)
        first = True

    layers = eval(args.layers)
    train_loader = DataLoader(cur_trainset, batch_size=args.batch_size,shuffle=True)
    model = MLP(cur_trainset, device, layers=layers).to(device)
    model.allgrad = []
    if first==True:
        model.allgrad = torch.zeros(int(args.epochs),len(cur_trainset)+len(val_loader.dataset)+len(test_loader.dataset),model.last_size)

    test_rmse,final_model = model_train_and_test(args, model, train_loader, val_loader, test_loader,first)

    del cur_trainset
    if new_tensor.shape[0]!=0:
        del new_tensor
    if new_val.shape[0]!=0:
        del new_val
    del model

    
    if first==True:
        print('[DONE] Step 1: tensor embedding learning')
        #Step 2: cell importance calculation
        train_idx,val_idx,test_idx = train_loader.dataset.indices,val_loader.dataset.indices,test_loader.dataset.indices
        checkpoint = final_model.checkpoint
        val_grad = torch.sum(final_model.allgrad[:checkpoint,val_idx,:],dim=1).squeeze()
        maxv,maxp = -9999,0
        final_model.importance = np.zeros(len(trainset))
        for (i,idx) in enumerate(trainset.indices):
            train_grad = final_model.allgrad[:checkpoint,idx,:].squeeze()
            contribution = torch.mul(train_grad,val_grad)
            final_contribution = torch.sum(torch.sum(contribution,dim=1),dim=0).item()
            final_model.importance[i] = final_contribution

        final_model.importance = final_model.importance / max(final_model.importance)

    return (test_rmse,final_model)

def main():

    args = parse_args()
    path = args.path
    layers = eval(args.layers)
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    output_path = 'output/'+args.output

    if os.path.exists('output/')==False:
        os.mkdir('output/')
    
    dataset = TensorDataset(path)
    trainset,valset, testset,indices = copy.deepcopy(dataset),copy.deepcopy(dataset),copy.deepcopy(dataset),np.arange(dataset.num_data)

    data_train, data_test, labels_train, labels_test, index_train, index_test = train_test_split(dataset.tensor.numpy(), dataset.val.numpy(), indices, test_size=1-args.train_ratio)
    data_train, data_val, labels_train, labels_val, index_train, index_val = train_test_split(data_train, labels_train, index_train, test_size=0.2)
    trainset.tensor,trainset.val,trainset.num_data,trainset.indices = torch.from_numpy(data_train).long(),torch.from_numpy(labels_train).float(),data_train.shape[0],torch.from_numpy(index_train).long()
    valset.tensor,valset.val,valset.num_data,valset.indices = torch.from_numpy(data_val).long(),torch.from_numpy(labels_val).float(),data_val.shape[0],torch.from_numpy(index_val).long()
    testset.tensor, testset.val, testset.num_data,testset.indices = torch.from_numpy(data_test).long(), torch.from_numpy(labels_test).float(), data_test.shape[0],torch.from_numpy(index_test).long()

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    print('[DONE] Step 0: Dataset loading & train-val-test split')
    print(dataset.dimensionality)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
 
    # CUDA for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #Step 1&2. Train tensor embeddings & calculate cell importance
    (rmse,model) = data_augmentation(trainset,torch.empty(0),torch.empty(0),val_loader,test_loader,args,device)
    print('Test RMSE before 50% data augmentation = {}'.format(rmse))
    print('Test RMSE before 50% data augmentation = {}'.format(rmse),file=open(output_path,"a"))
    original = copy.deepcopy(model)
    del model

    cell_importance = abs(original.importance)
    print('[DONE] Step 2: cell importance calculation')

    #Step 3. entity importance calculation
    entity_importance = [np.zeros(dataset.dimensionality[i]) for i in range(dataset.order)]
    for i in range(len(cell_importance)):
        for j in range(dataset.order):
            entity = int(trainset.tensor[i,j])
            entity_importance[j][entity] += cell_importance[i]

    for i in range(dataset.order):
        cur = entity_importance[i]
        entity_importance[i] = cur/sum(cur)
    
    print('[DONE] Step 3: entity importance calculation')

    num_aug = int(0.5 * trainset.tensor.shape[0])

    print('Number of augmented data = {}\tTotal number of training data = {}'.format(num_aug,num_aug+len(trainset)))
    print('Number of augmented data = {}\tTotal number of training data = {}'.format(num_aug,num_aug+len(trainset)), file=open(output_path, "a"),flush=True)

    #Step 4. perform data augmentation
    indices = np.zeros((num_aug,trainset.order))
    for i in range(dataset.order):
        indices[:,i] =  np.random.choice(list(range(0,dataset.dimensionality[i])),size=num_aug,p = entity_importance[i])
    new_tensor = torch.from_numpy(indices).long()
    new_val = original.predict(new_tensor)
    print('[DONE] Step 4: data augmentation with entity importance')
    (rmse,model) = data_augmentation(trainset,new_tensor,new_val,val_loader,test_loader,args,device)
    print('Test RMSE after 50% data augmentation = {}'.format(rmse))
    print('Test RMSE after 50% data augmentation = {}'.format(rmse),file=open(output_path,"a"))
    del model

if __name__ == "__main__":
        main()

