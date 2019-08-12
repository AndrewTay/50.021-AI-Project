import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
train_on_gpu = True
num_epochs = 8
num_classes = 2
batch_size = 128
learning_rate = 0.1

# Device configuration
device = torch.device('cuda:0')

labels = pd.read_csv('data/train_labels.csv')
train_path = 'data/train/'
test_path = 'data/test/'

print(f'{len(os.listdir(train_path))} pictures in train.')
print(f'{len(os.listdir(test_path))} pictures in test.')


class MyDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.tif')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train, val = train_test_split(labels, stratify=labels.label, test_size=0.1)

trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)
train_loader = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)


def train(model, loss_fn, device, train_loader, optimizer):
	#training of model
    model.train()
    samples_trained = 0
    training_loss = 0
    #for data set 
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        samples_trained += data.size()[0]
        print('Trained %d samples, loss: %10f' %(samples_trained, training_loss/samples_trained), end="\r")
        del data
        del target
    training_loss /= samples_trained
    return training_loss

def test(model, loss_fn, device, test_loader):
	#evaluating model on validation/test dataset
    model.eval()
    correct_pred = 0
    samples_tested = 0
    test_loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct_pred += torch.sum(pred==target).item()
            samples_tested += data.size()[0]
            print('Tested %d samples, loss: %10f' %(samples_tested, test_loss/samples_tested), end="\r")
    accuracy = correct_pred/samples_tested
    test_loss /= samples_tested
    return accuracy, test_loss

def train_val_test(model, loss_fn, device, optimizer, epoch, name):
    #function to train model, valdidate data against validation set and finally running the model on the test dataset
    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    highest_val_loss = 1
    test_loss_list = []

    #training and validating across epochs
    for t in range(epoch):
        print ('Current epoch: ', t+1)
        training_loss = train(model, loss_fn, device, train_loader, optimizer)
        print ('')
        accuracy, val_loss = test(model, loss_fn, device, valid_loader)
        print ('')
        if val_loss < highest_val_loss:
            highest_val_loss = val_loss
            torch.save(model.state_dict(), name + '.pt')
            
        val_accuracy_list.append(accuracy)
        training_loss_list.append(training_loss)
        val_loss_list.append(val_loss)

    #plot and save graph
    plt.plot(val_loss_list, label='val loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_val_loss.png')
    plt.clf()

    plt.plot(training_loss_list, label='training loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_training_loss.png')
    plt.clf()

    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend(loc='upper left')
    plt.savefig(name + '.png')
    plt.clf()

    #torch.save(model.state_dict(), 'model.ckpt')
    #run on test set
    model.load_state_dict(torch.load(name + '.pt'))
    test_accuracy, test_loss = test(model, loss_fn, device, test_loader)
    test_loss_list.append(test_loss)
    print ("Accuracy is: ", test_accuracy)
    with open(name + '.txt', 'w') as txtfile:
        txtfile.write("Accuracy is: "+ str(test_accuracy))
    plt.plot(test_loss_list, label ='test loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_test_loss.png')
    plt.clf()

isdev = True

torch.cuda.empty_cache()
device = torch.device('cuda:0')
loss_fn = F.cross_entropy
epoch = 1 if isdev else 8
learningrate = 0.01
model3 = models.inception_v3(pretrained=True)
optimizer = torch.optim.Adamax(model3.parameters(), lr=learning_rate)
model3.to(device)
train_val_test(model3, loss_fn, device, optimizer, epoch, 'model3')

  

