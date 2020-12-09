"""
Self-organized Gragh Neural Network
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datapipe import build_dataset, get_dataset
from Net import SOGNN

##########################################################
"""
Settings for training
"""
subjects = 15
epochs = 200
classes = 3 # Num. of classes 
Network = SOGNN

version = 1
print('***'*20)
while(1):
    dfile = './result/{}_LOG_{:.0f}.csv'.format(Network.__name__, version)
    if not os.path.exists(dfile):
        break
    version += 1
print(dfile)
df = pd.DataFrame()
df.to_csv(dfile)   
##########################################################

def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader: 
        data = data.to(device)
        optimizer.zero_grad()
        
        #Multiple Classes classification Loss function
        label = torch.argmax(data.y.view(-1,classes), axis=1)
        label = label.to(device)#, dtype=torch.long) #, dtype=torch.int64)
        
        output, _ = model(data.x, data.edge_index, data.batch)
        
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step() 

    return loss_all / len(train_dataset)

def evaluate(model, loader, save_result=False):
    model.eval()

    predictions = []
    labels = [] 

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1,classes)
            data = data.to(device)
            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy() 
            
            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    #AUC score estimation 
    AUC = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    #Accuracy 
    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    acc = accuracy_score(labels, predictions)

    return AUC, acc, f1
    
build_dataset(subjects)# Build dataset for each fold

print('Cross Validation')
result_data = []
all_last_acc = []
all_last_AUC = []
for cv_n in range(0, subjects): 
    train_dataset, test_dataset = get_dataset(subjects, cv_n)
    train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True )
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('cuda', 0)
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    crit = torch.nn.CrossEntropyLoss() # 
    
    epoch_data = []
    for epoch in range(epochs):
        t0 = time.time()
        loss = train(model, train_loader, crit, optimizer)
        train_AUC, train_acc, train_f1 = evaluate(model, train_loader)
        val_AUC, val_acc, val_f1 = evaluate(model, test_loader)

        epoch_data.append([str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc])
        t1 = time.time()
        print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))
        if train_AUC>0.99 and train_acc>0.90:
            break

    print('Results::::::::::::')
    print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))
    
    result_data.append([str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, val_f1])
    
    df = pd.DataFrame(data=result_data, columns=['Fold', 'Epoch', 'Loss', 'Tra_AUC', 'Tra_acc', 'Val_AUC', 'Val_acc', 'Val_f1'])
    
    df.to_csv(dfile)  
    
df = pd.read_csv(dfile)

lastacc = ['Val_acc', df['Val_acc'].mean()]
lastauc = ['Val_AUC', df['Val_AUC'].mean()] 
print(lastacc) 
print(lastauc) 
print('*****************')


