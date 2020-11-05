from flask import Flask, render_template,request, jsonify
import pandas as pd
import numpy as np
from utils import DogBreedClassifier
from PIL import Image,ImageFile
import os
import boto3


application = Flask(__name__, static_folder='static')

s3 = boto3.client('s3',
aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
aws_secret_access_key=os.environ['AWS_SECRET_KEY']
)

if(os.path.isfile('LabelID_DF.pickle')==False):
    s3.download_file('sh-apps-bucket', 'dogApp/LabelID_DF.pickle', 'LabelID_DF.pickle')
if(os.path.isfile('model_transfer_CPU.pickle')==False):
    s3.download_file('sh-apps-bucket', 'dogApp/model_transfer_CPU.pickle', 'model_transfer_CPU.pickle')



dogCLF=DogBreedClassifier()


@application.route('/', methods=['GET','POST'])
def homepage(): 
    return render_template('app.html')

@application.route('/imageUpload', methods=['GET','POST'])
def imageUpload():
    content={}
    if(request.method=="POST"):
        print(request.files)
        imagefile=request.files.get('fileUpload')
        testimage = Image.open(imagefile)
        out=dogCLF.getPreds(testimage)
        breeds=dogCLF.getBreeds(out)
        #print(breeds)
        #return render_template('app.html')
        content={'pred1':breeds[0],'pred2':breeds[1],'pred3':breeds[2]}
        #content={'pred1':'dog1','pred2':'dog2','pred3':'dog3'}
    return jsonify(content)
    #return render_template('app.html',data=content)

if __name__=="__main__":
    application.run(host='0.0.0.0')