

from IPython import get_ipython
import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)

import mxnet as mx
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)


mod = mx.mod.Module(symbol=sym, context=mx.cpu())


# The ResNet is trained with RGB images of size 224 x 224. The training data is feed by the variable `data`. We bind the module with the input shape and specify that it is only for predicting. The number 1 added before the image shape (3x224x224) means that we will only predict one image each time. Next we set the loaded parameters. Now the module is ready to run. 


mod.bind(for_training = False,
         data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)


# We first obtain the synset file, in which the i-th line contains the label for the i-th class.

download('http://data.mxnet.io/models/imagenet/resnet/synset.txt')
with open('synset.txt') as f:
    synsets = [l.rstrip() for l in f]


import matplotlib
import matplotlib.pyplot as plt
import cv2

# Next we define a function that reads one image each time and convert to a format can be used by the model. Here we use a naive way that resizes the original image into the desired shape, and change the data layout. 

import numpy as np
import cv2
def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
    img = img[np.newaxis, :]  # extend to (example, channel, heigth, width)
    return img


# Finally we define a input data structure which is acceptable by mxnet. The field `data` is used for the input data, which is a list of NDArrays. 


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])



# ## Extract Features
# 
# The neural network works as a feature extraction module to other applications. 
# 
# A loaded symbol in default only returns the last layer as output. But we can get all internal layers by `get_internals`, which returns a new symbol outputting all internal layers. The following codes print the last 10 layer names. 
# 

all_layers = sym.get_internals()
all_layers.list_outputs()[-10:-1]


# Often we want to use the output before the last fully connected layers, which may return semantic features of the raw images but not too fitting to the label yet. In the ResNet case, it is the flatten layer with name `flatten0` before the last fullc layer. The following codes get the new symbol `sym3` which use the flatten layer as the last output layer, and initialize a new module.


all_layers = sym.get_internals()
sym3 = all_layers['flatten0_output']
mod3 = mx.mod.Module(symbol=sym3, context=mx.cpu())
mod3.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod3.set_params(arg_params, aux_params)


# Now we can do feature extraction using `forward1` as before. Notice that the last convolution layer uses 2048 channels, and we then perform an average pooling, so the output size of the flatten layer is 2048.



#extraction of features of our classes
from PIL import Image
import glob


'''To extract features of dataset using ResNet-50 CNN'''
# football = np.array([get_image('football/' + img) for img in os.listdir( 'football/')])
# lion = np.array([get_image('lion/' + img) for img in os.listdir('lion/')])
# guitar = np.array([get_image( 'guitar/' + img) for img in os.listdir('guitar/')])
# art = np.array([get_image( 'art/' + img) for img in os.listdir( 'art/')])
# buildings = np.array([get_image( 'buildings/' + img) for img in os.listdir('buildings/')])
# people = np.array([get_image( 'people/' + img) for img in os.listdir( 'people/')])
# piano = np.array([get_image( 'piano/' + img) for img in os.listdir( 'piano/')])
# statues = np.array([get_image( 'statues/' + img) for img in os.listdir('statues/')])
# all_images = np.vstack([football,lion,guitar,art,buildings,people,piano,statues])
# all_images = np.array(datas)

'''To extract features of queries using ResNet-50 CNN'''
datas = np.array([get_image('queryfolder/'+img) for img in os.listdir('queryfolder/')])


import pandas as pd



q_features =[]

from sklearn.decomposition import SparsePCA 

for i in range(len(datas)):
    mod3.forward(Batch([mx.nd.array(datas[i])]))
    pred = (mod3.get_outputs()[0].asnumpy())
    pred = pred.reshape(-1)
    q_features.append(pred)

q_features = np.array(q_features)

df = pd.DataFrame(q_features)
'''Store the features of queries by CNN method'''
df.to_csv('query_values_full.csv')
