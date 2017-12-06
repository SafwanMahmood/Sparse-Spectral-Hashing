

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

# In[19]:

mod.bind(for_training = False,
         data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)


# ## Prepare data
# 
# We first obtain the synset file, in which the i-th line contains the label for the i-th class.

# In[20]:

download('http://data.mxnet.io/models/imagenet/resnet/synset.txt')
with open('synset.txt') as f:
    synsets = [l.rstrip() for l in f]


# We next download 1000 images for testing, which were not used for the training. 

# In[21]:


# Visualize the first 8 images.

# In[22]:

# get_ipython().magic(u'matplotlib inline')
import matplotlib
# matplotlib.rc("savefig", dpi=100)
import matplotlib.pyplot as plt
import cv2

# Next we define a function that reads one image each time and convert to a format can be used by the model. Here we use a naive way that resizes the original image into the desired shape, and change the data layout. 

# In[23]:

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

# In[24]:

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


# ## Predict
# 
# Now we are ready to run the prediction by `forward`. Then we can get the output using `get_outputs`, in which the i-th element is the predicted probability that the image contains the i-th class. 

# In[25]:

# img = get_image('val_1000/0.jpg')
# mod.forward(Batch([mx.nd.array(img)]))
# prob = mod.get_outputs()[0].asnumpy()
# y = np.argsort(np.squeeze(prob))[::-1]
# print('truth label %d; top-1 predict label %d' % (val_label[0], y[0]))


# When predicting more than one images, we can batch several images together which potentially improves the performance. 

# In[26]:

# batch_size = 32
# mod2 = mx.mod.Module(symbol=sym, context=mx.cpu())
# mod2.bind(for_training=False, data_shapes=[('data', (batch_size,3,224,224))])
# mod2.set_params(arg_params, aux_params)


# Now we iterative multiple images to calculate the accuracy

# In[27]:

# Output may vary
# import time
# acc = 0.0
# total = 0.0
# for i in range(0, 200/batch_size):
#     tic = time.time()
#     idx = range(i*batch_size, (i+1)*batch_size)
#     img = np.concatenate([get_image('val_1000/%d.jpg'%(j)) for j in idx])
#     mod2.forward(Batch([mx.nd.array(img)]))
#     prob = mod2.get_outputs()[0].asnumpy()
#     pred = np.argsort(prob, axis=1)
#     top1 = pred[:,-1]
#     acc += sum(top1 == np.array([val_label[j] for j in idx]))
#     total += len(idx)
#     print('batch %d, time %f sec'%(i, time.time()-tic))
# assert acc/total > 0.66, "Low top-1 accuracy."
# print('top-1 accuracy %f'%(acc/total))


# ## Extract Features
# 
# The neural network works as a feature extraction module to other applications. 
# 
# A loaded symbol in default only returns the last layer as output. But we can get all internal layers by `get_internals`, which returns a new symbol outputting all internal layers. The following codes print the last 10 layer names. 
# 
# We can also use `mx.viz.plot_network(sym)` to visually find the name of the layer we want to use. The name conventions of the output is the layer name with `_output` as the postfix.

# In[28]:

all_layers = sym.get_internals()
all_layers.list_outputs()[-10:-1]


# Often we want to use the output before the last fully connected layers, which may return semantic features of the raw images but not too fitting to the label yet. In the ResNet case, it is the flatten layer with name `flatten0` before the last fullc layer. The following codes get the new symbol `sym3` which use the flatten layer as the last output layer, and initialize a new module.

# In[29]:

all_layers = sym.get_internals()
sym3 = all_layers['flatten0_output']
mod3 = mx.mod.Module(symbol=sym3, context=mx.cpu())
mod3.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod3.set_params(arg_params, aux_params)


# Now we can do feature extraction using `forward1` as before. Notice that the last convolution layer uses 2048 channels, and we then perform an average pooling, so the output size of the flatten layer is 2048.

# We now add the classes we want to classify on. Here I added football and lion class.  

# In[30]:

#extraction of features of our classes
from PIL import Image
import glob

# out=[]
# data_pos = np.array([get_image('football/'+img) for img in os.listdir('football/')])
# data_neg = np.array([get_image('lion/'+img) for img in os.listdir('lion/')])
# # data_pos1 = np.array([get_image('piano/'+img) for img in os.listdir('piano/')])
# data_neg1 = np.array([get_image('guitar/'+img) for img in os.listdir('guitar/')])
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

# # print q_features,q_features.shape
# pca = SparsePCA(n_components=1024)
# x = pca.fit_transform(q_features[0].reshape(1,-1))

# print x
df = pd.DataFrame(q_features)
# # df = df.drop(df.columns[0], axis=0)
df.to_csv('query_values_full.csv')
# ## Further Readings
# 
