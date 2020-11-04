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
#torch.cuda.get_device_name(0)

"""
testImgPath="C:\\Users\\shard\\Pictures\\Otis.JPEG"
testtransform = transforms.Compose([transforms.Resize(256),
                                    #transforms.RandomResizedCrop(224), 
                                    transforms.ToTensor()
    ])
# Load image and run through transform      
testimage = testtransform(Image.open(testImgPath))
testimage
testimage=testimage.unsqueeze(0)
model_transfer = models.resnet18(pretrained=True)
if use_cuda:
    model_transfer=model_transfer.cuda()
model_transfer.eval()
if use_cuda:
    out=model_transfer(testimage.cuda()).cpu()
else:
    out=model_transfer(testimage)
out
out=torch.argmax(torch.nn.functional.softmax(out,dim=1))
print(out)
"""

#Load in model transfer
model_transfer=pickle.load(open('model_transfer.pickle','rb'))
model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))
model_transfer.eval()
testImgPath="C:\\Users\\shard\\Pictures\\Otis.JPEG"
labelID_DF=pickle.load(open('LabelID_DF.pickle','rb'))
testtransform = transforms.Compose([transforms.Resize(256),
                                    #transforms.RandomResizedCrop(224), 
                                    transforms.ToTensor()
    ])
# Load image and run through transform      
testimage = testtransform(Image.open(testImgPath))
# Add dimension to tensor for number of images
testimage = testimage.unsqueeze(0)
model_transfer.eval()
model_transfer.cpu()
if use_cuda:
        # We can now make our prediction by passing the image_variable through the
        # VGG16 network to get a prediction.
        prediction = model_transfer(Variable(testimage).cuda()).cpu()
else:
        # We can now make our prediction by passing the image_variable through the
        # VGG16 network to get a prediction.
        prediction = model_transfer(Variable(testimage))
prediction
prediction=prediction.data.numpy().argmax()
print(labelID_DF.loc[prediction]['breed'])



import pickle
import boto3
from utils import DogBreedClassifier

s3 = boto3.resource('s3',
aws_access_key_id='LOOK IT UP!!',
aws_secret_access_key='LOOK IT UP!!!'
)

s3=boto3.resource('s3')

#model_transfer = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/model_transfer.pickle").get()['Body'].read())
#model_transfer
torch.hub.download_url_to_file('https://sh-apps-bucket.s3.amazonaws.com/dogApp/model_transfer.pickle',"C:\\Users\\shard\\Documents\\PythonProjects\\DogBreed\\MODEL.pickle")
model_transfer=pickle.load(open('MODEL.pickle','rb'))
model_transfer.eval()
model_transfer.cpu()
model_transfer.eval()

with open('model_transfer_CPU.pickle', 'wb') as handle:
    pickle.dump(model_transfer,handle)

model_transfer=pickle.load(open('model_transfer_CPU.pickle','rb'))
#Try reading from S3
model_transfer = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/model_transfer_CPU.pickle").get()['Body'].read())
model_transfer

labelID_DF = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/LabelID_DF.pickle").get()['Body'].read())
labelID_DF

###################################################################
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

import pickle
import boto3
from utils import DogBreedClassifier


s3 = boto3.resource('s3',
aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
aws_secret_access_key=os.environ['AWS_SECRET_KEY']
)

s3=boto3.resource('s3',verify=False)

model_transfer = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/model_transfer_CPU.pickle").get()['Body'].read())
model_transfer

labelID_DF = pickle.loads(s3.Bucket("sh-apps-bucket").Object("dogApp/LabelID_DF.pickle").get()['Body'].read())
labelID_DF

testImgPath="C:\\Users\\shard\\Pictures\\Otis.JPEG"
testtransform = transforms.Compose([transforms.Resize(256),
                                    #transforms.RandomResizedCrop(224), 
                                    transforms.ToTensor()
    ])
# Load image and run through transform      
testimage = testtransform(Image.open(testImgPath))
# Add dimension to tensor for number of images
testimage = testimage.unsqueeze(0)
model_transfer.eval()
prediction = model_transfer(Variable(testimage))
prediction
prediction=prediction.data.numpy().argmax()
print(labelID_DF.loc[prediction]['breed'])



session=boto3.session.Session()
s3client=session.client('s3')
response=s3client.get_object(Bucket='sh-apps-bucket',Key='dogApp/LabelID_DF.pickle')
