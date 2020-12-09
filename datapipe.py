"""

""" 
version = 1

import os 
import numpy as np 
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import scipy.io as sio
import glob  

predictions_dir = './predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
             
subjects = 15 # Num. of subjects used for LOSO
classes = 3 # Num. of classes 

def to_categorical(y, num_classes=None, dtype='float32'): 
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None):
        self.stage = stage #Train or test
        self.subjects = subjects  
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        
        #super(EmotionDataset, self).__init__(root, transform, pre_transform)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, self.stage, self.subjects, self.sub_i)]
    def download(self):
        pass
    
    def process(self): 
        data_list = [] 
        # process by samples
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)): 
            x = self.X[sample_id,:,:]    
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(self.Y[sample_id,:])
            data = Data(x=x, y=y)

            data_list.append(data) 
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data 

def get_data():
    path = './emotion_data/SEED/ExtractedFeatures/'
    label = sio.loadmat(path+'label.mat')['label']
    files = sorted(glob.glob(path+'*_*'))

    sublist = set()
    for f in files:
        sublist.add(f.split('/')[-1].split('_')[0] )
    
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    sublist = sorted(list(sublist))
    print(sublist)
   
    sub_mov = [] 
    sub_label = []
    
    for sub_i in range(subjects):
        sub = sublist[sub_i]
        sub_files = glob.glob(path+sub+'*')
        mov_data = [] 
        for f in sub_files:
            print(f)
            data = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = data.keys()
            de_mov = [k for k in keys if 'de_movingAve' in k] 
        
            mov_datai = [] 
            for t in range(15):
                temp_data = data[de_mov[t]].transpose(0,2,1)
                data_length  = temp_data.shape[-1]
                mov_i = np.zeros((62, 5, 265))
                mov_i[:,:,:data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)#.transpose(1,0)
                
                mov_datai.append(mov_i) 
            mov_datai = np.array(mov_datai)  
            mov_data.append(mov_datai) 
            
        mov_data = np.vstack(mov_data) 
        mov_data = normalize(mov_data) 
        sub_mov.append(mov_data)
        sub_label.append(np.hstack([label, label, label]).squeeze())
        
    sub_mov = np.array(sub_mov) 
    sub_label = np.array(sub_label)

    return sub_mov, sub_label
    
def build_dataset(subjects):
    load_flag = True
    for sub_i in range(subjects):
        path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, 'Train', subjects, sub_i)
        print(path)
        
        if not os.path.exists(path): 
        
            if load_flag:
                mov_coefs, labels = get_data()
                used_coefs = mov_coefs
                load_flag = False
            
            index_list = list(range(subjects))
            del index_list[sub_i]
            test_index = sub_i
            train_index = index_list
            
            print('Building train and test dataset')
            #get train & test
            X = used_coefs[train_index,:].reshape(-1, 62, 265*5)
            Y = labels[train_index,:].reshape(-1)
            testX = used_coefs[test_index,:].reshape(-1, 62, 265*5)
            testY = labels[test_index,:].reshape(-1) 
            #get labels
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)#
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)
            
            train_dataset = EmotionDataset('Train', './', subjects, sub_i, X, Y)
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY)
            print('Dataset is built.')
            
def get_dataset(subjects, sub_i):
    path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
            version, 'Train', subjects, sub_i)
    print(path)
    if not os.path.exists(path): 
        raise IOError('Train dataset is not exist!')
    
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)
    test_dataset = EmotionDataset('Test', './', subjects, sub_i) 

    return train_dataset, test_dataset
