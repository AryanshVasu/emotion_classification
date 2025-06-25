import numpy as np
import torch
from torch import nn
import os
import matplotlib.pyplot as plt
import librosa
import csv

from sklearn.metrics import accuracy_score

# function to get features from URL
def transform_pipeline(url):
    sig, sr = librosa.load(url,duration=3,offset=0.9)

    # clipping to const duration
    maxSamples = 3*sr 
    n = len(sig)
    if (n > maxSamples):
        sig = sig[:maxSamples]
    elif (n < maxSamples):
        padding = np.zeros(maxSamples-n)
        sig = np.concatenate((sig,padding))
    
    # spectral 
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_fft=2048, hop_length=512, n_mfcc=32)
    features = torch.stack([torch.FloatTensor(mfcc)])
    return features

# custom dataset
class AudioData(torch.utils.data.Dataset):
    def __init__(self):
        dataURL = './eval_data/'

        self.labels = [int(audio.split('-')[2])-1 for audio in os.listdir(dataURL)]
        self.URLs = [dataURL + audio for audio in os.listdir(dataURL)]

        self.labels = np.array(self.labels)
        self.URLs = np.array (self.URLs)
    
    def __getitem__(self,idx):
        audioURL = self.URLs[idx]
        features = transform_pipeline(audioURL)
        return features, torch.tensor(self.labels[idx])
    
    def __len__(self):
        return len(self.URLs)

evalDS = AudioData()

print (f'> Running evaluatin on {len(evalDS)} audio samples :\n')

# model 2
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        # common layers
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(2,2)
        # convolution layers
        self.conv1 = nn.Conv2d(1,8,3,1,1)
        self.bnorm1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8,16,3,1,1)
        self.bnorm2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16,32,3,1,1)
        self.bnorm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32,16,3,1,1)
        self.bnorm4 = nn.BatchNorm2d(16)
        # fully connected layer
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16*2*8,8)

    def forward(self, x):
        # x,y = X
        x = self.bnorm1(self.act(self.conv1(x)))
        x = self.pool(x)
        x = self.bnorm2(self.act(self.conv2(x)))
        x = self.pool(x)
        x = self.bnorm3(self.act(self.conv3(x)))
        x = self.pool(x)
        x = self.bnorm4(self.act(self.conv4(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x 
        
evalModel = CNN2()
evalModel.load_state_dict((torch.load('model.pt')))

testLoader = torch.utils.data.DataLoader(evalDS,batch_size=len(evalDS))
with torch.no_grad():
    evalModel.to('cpu')
    features, labels = next(iter(testLoader))
    features = features.to('cpu')
    labels = labels.to('cpu')
    outputs = evalModel(features)
    _,pred = torch.max(outputs,1)
    # report = classification_report(labels,pred)

print (f'> Evaluation completed with accuracy = {accuracy_score(labels,pred)*100 :.2f} \nOutput stored in out.csv.\n')

out = {}
pred = pred.numpy()
emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']


for i, url in enumerate(evalDS.URLs):
    out[url] = emotions[pred[i]]


with open("out.csv", "w", newline="") as f:
    w = csv.DictWriter(f, out.keys())
    w.writeheader()
    w.writerow(out)


