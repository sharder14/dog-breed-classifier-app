import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset,Dataset
from PIL import Image,ImageFile
from collections import OrderedDict
import numpy as np
from glob import glob
import os
import pandas as pd
import numpy as np
import pickle
#use_cuda = torch.cuda.is_available()
use_cuda=False
#torch.cuda.get_device_name(0)
import boto3

#Load in model transfer
class DogBreedClassifier():

    def __init__(self):
        '''
        s3 = boto3.resource('s3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
            aws_secret_access_key=os.environ['AWS_SECRET_KEY']
        )
        '''
        #self.model_transfer=pickle.load(open('model_transfer.pickle','rb'))
        #self.model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))
        #self.model_transfer = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/model_transfer_CPU.pickle").get()['Body'].read())
        #self.labelID_DF = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/LabelID_DF.pickle").get()['Body'].read())
        self.model_transfer=pickle.load(open('model_transfer_CPU.pickle','rb'))
        self.labelID_DF=pickle.load(open('LabelID_DF.pickle','rb'))

        
        self.model_transfer.eval()
        #self.model_transfer.cpu()
        #self.labelID_DF=pickle.load(open('LabelID_DF.pickle','rb'))
        
        self.testtransform = transforms.Compose([transforms.Resize(256),
                                    #transforms.RandomResizedCrop(224), 
                                    transforms.ToTensor()
        ])

    def getPreds(self,imageTensor):
        testimage=self.testtransform(imageTensor)
        testimage = testimage.unsqueeze(0)
        if use_cuda:
                # We can now make our prediction by passing the image_variable through the
                # VGG16 network to get a prediction.
                prediction = self.model_transfer(Variable(testimage).cuda()).cpu()
        else:
                # We can now make our prediction by passing the image_variable through the
                # VGG16 network to get a prediction.
                prediction = self.model_transfer(Variable(testimage))
        return prediction

    def getBreeds(self,predTensor):
        vals, inds= torch.sort(predTensor)
        out=self.labelID_DF.loc[inds[0][len(inds[0])-3:].detach().numpy(),'breed'].values[::-1]
        return out

'''
dogCLF=DogBreedClassifier()
testImgPath="C:\\Users\\shard\\Pictures\\Otis.JPEG"
# Load image and run through transform      
testimage = Image.open(testImgPath)
out=dogCLF.getPreds(testimage)
out
dogCLF.getBreeds(out)

vals, inds=torch.sort(out)
dogCLF.labelID_DF.loc[inds[0][len(inds[0])-3:].detach().numpy(),'breed'].values[::-1]
'''