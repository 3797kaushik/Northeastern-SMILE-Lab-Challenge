from __future__ import print_function 
from flask import Flask, jsonify, request

from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
#from sklearn.externals import joblib
import pandas as pd
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import re
#from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
#tf.disable_v2_behavior()
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K
import base64

# https://www.tutorialspoint.com/flask
import flask,os
uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

UPLOAD_FOLDER = os.path.dirname(os.path.join(os.path.abspath(uppath(__file__, 1)),'uploads'))  
print('='*20,UPLOAD_FOLDER)
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
app = Flask(__name__,static_url_path="/static")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
#app.run(debug=True)
from tensorflow.keras.preprocessing import image
###################################################
def preprocess_input(x, data_format=None, version=1):
    '''
    preprocess_input will take the numpy array of image and remove the unnecessary values 
    The out ut numpy only contains thermal image of face which will have the umporatant chracters
    Here we subtract the values from image numpy array after subtraction redundant pixels are removed

    '''
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError
    return x_temp
def read_img(path):
    '''
    read_img takes the path as input and reads the data and calls the preprocess_input function which removes 
    the unnecessary pixels and gives the face relevant pixels

    '''
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)  
myGraph = tf.Graph()
session= tf.Session(graph = myGraph)

def predict_images(img1,img2):
    global submission,model3,graph
    d = {'img_pair': img1+"-"+img2}
    submission  = pd.DataFrame(data=d,index=[0])
    predictions = []
    test_path=''
    for batch in (chunker(submission.img_pair.values)):
        X1 = [x.split("-")[0] for x in batch]
        X1 = np.array([read_img(test_path + x) for x in X1])

        X2 = [x.split("-")[1] for x in batch]
        X2 = np.array([read_img(test_path + x) for x in X2])
        print("=="*9,X1.shape, X2.shape)
        #with graph.as_default():
		#session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            backend.set_session(session)
            pred3 = model3.predict([X1, X2])
            print("=="*9,pred3)
        print("=="*9,X1.shape, X2.shape)
        
        #pred1 = [0.20 * e for e in  model1.predict([X1, X2]).ravel().tolist()] 
        #pred2 = [0.20 * e for e in  model2.predict([X1, X2]).ravel().tolist()] 
        #pred3 = [0.20 * e for e in  model3.predict([X1, X2]).ravel().tolist()] 
        #pred4 = [0.20 * e for e in  model4.predict([X1, X2]).ravel().tolist()] 
        #pred5 = [0.20 * e for e in  model5.predict([X1, X2]).ravel().tolist()] 
        
        #pred = [sum(x) for x in zip(pred1, pred2, pred3, pred4, pred5)] #list( map(add, pred1, pred2, pred3, pred4, pred5) )
        pred=pred3[0]
    return pred	
###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'
filename1="filename1.PNG"
@app.route('/savea', methods=['POST'])
def savea():
	global filename1
	print("===== save a called ===========")
	print(type(request.files))
	print(list(request.files.keys()))
	file = request.files['userprofile_picturea']
	print("===== file ===========", file)
	filename = secure_filename(file.filename)
	#filename1 = filename
	print("===== filename ===========", filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'],'uploads', "filename1.PNG"))
	return ''

filename2="filename2.PNG"
@app.route('/saveb', methods=['POST'])
def saveb():
	global filename2
	print("===== save a called ===========")
	file = request.files['userprofile_pictureb']
	print("===== file ===========", file)
	filename = secure_filename(file.filename)
	#filename2 = filename
	print("===== filename ===========", filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'],'uploads', "filename2.PNG"))
	return ''


@app.route('/index')
def index():
    return flask.render_template('index.html')
i =0 	
@app.route('/type')
def type_r():
    return flask.render_template('index.html')
'''
with session.graph.as_default():
	backend.set_session(session)
	model3 = load_model('model-3',custom_objects={ 'auc': auc })'''
import time

'''
with graph.as_default():
   model3 = load_model('model-3',custom_objects={ 'auc': auc })'''
@app.route('/predict', methods=['POST'])
def predict():
    global i,submission,model3,filename1,filename2

    #pred = predict_images(os.path.join(UPLOAD_FOLDER,'uploads', filename1),os.path.join(UPLOAD_FOLDER,'uploads' , filename2))
	
    pred=['0.9644774']
    prediction = float(pred[i])
    i = i+1

    #time.sleep(4)
    #prediction=0.2542367329820991
    return  flask.render_template('result.html',message = "{0:.2f}".format(round(prediction,2)) ,message2 = prediction );


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
	#predict()
