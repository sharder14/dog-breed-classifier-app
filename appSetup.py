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


s3 = boto3.client('s3',
aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
aws_secret_access_key=os.environ['AWS_SECRET_KEY']
)
s3.download_file('sh-apps-bucket', 'dogApp/model_transfer_CPU.pickle', 'model_transfer_CPU.pickle')
s3.download_file('sh-apps-bucket', 'dogApp/LabelID_DF.pickle', 'LabelID_DF.pickle')

"""
model_transfer=pickle.load(open('model_transfer_CPU.pickle','rb'))
model_transfer
labelID_DF=pickle.load(open('LabelID_DF.pickle','rb'))
labelID_DF
"""
#os.path.isfile('LabelID_DF.pickle')