
'''Sparse Spectral Hashing'''
'''Author Safwan Mahmood, Aashish Patole, Sanjan Prakash, Vibhas Goyal'''


from sklearn.decomposition import SparsePCA 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd 
import pylab
import os
from PIL import Image
import glob
import math
import scipy as sc
import itertools

import boosting  as bb


'''Preprocess the image'''
def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    return img


'''Taking input of the dataset'''
football = np.array([get_image('football/' + img) for img in os.listdir( 'football/')])
lion = np.array([get_image('lion/' + img) for img in os.listdir('lion/')])
guitar = np.array([get_image( 'guitar/' + img) for img in os.listdir('guitar/')])
art = np.array([get_image( 'art/' + img) for img in os.listdir( 'art/')])
buildings = np.array([get_image( 'buildings/' + img) for img in os.listdir('buildings/')])
people = np.array([get_image( 'people/' + img) for img in os.listdir( 'people/')])
piano = np.array([get_image( 'piano/' + img) for img in os.listdir( 'piano/')])
statues = np.array([get_image( 'statues/' + img) for img in os.listdir('statues/')])

all_images = np.vstack([football,lion,guitar,art,buildings,people,piano,statues])
all_images = np.array(all_images)

'''Labels for the categories for boosting algo'''
n = 0
labels = []
for i in range(0,n + len(football)):
	labels.append(1)
n += len(football)
for i in range(n,n + len(lion)):
	labels.append(2)
n += len(lion)
for i in range(n,n + len(guitar)):
	labels.append(3)
n += len(guitar)
for i in range(n,n + len(art)):
	labels.append(4)
n += len(art)
for i in range(n,n + len(buildings)):
	labels.append(5)
n += len(buildings)
for i in range(n,n + len(people)):
	labels.append(6)
n += len(people)
for i in range(n,n + len(piano)):
	labels.append(7)
n += len(piano)
for i in range(n,n + len(statues)):
	labels.append(8)
n += len(statues)

labels.append(8)

'''Store the labels in CSV'''
df = pd.DataFrame(labels)
df.to_csv('labels.csv')





		


'''Below contains the csv file of features extracted by various methods for the images.'''
''' No need to uncomment as we have cached the final Y-values for them'''

'''Featurelist for CNN'''
# X = pd.read_csv('Featurelist.csv', delimiter=None)     								#All images after preprocessing								
# X = X.drop(X.columns[0], axis=1)
# X = np.array(X)
'''pca with 1024 components'''
# pca = SparsePCA(n_components=1024)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('x_features_after_pca_1024.csv')
'''pca with 512 components'''
# pca = SparsePCA(n_components=512)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('x_features_after_pca_512.csv')
'''pca with 256 components'''
# pca = SparsePCA(n_components=256)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('x_features_after_pca_256.csv')

'''Featurelist for BRIEF'''
# X = pd.read_csv('feature_list_heavy.csv', delimiter=None)								#All images after preprocessing								
# X = X.drop(X.columns[0], axis=1)
# X = np.array(X)
'''pca with 16 components'''
# pca = SparsePCA(n_components=16)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('heavy_components_16.csv')
'''pca with 8 components'''
# pca = SparsePCA(n_components=8)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('heavy_components_8.csv')
'''pca with 4 components'''
# pca = SparsePCA(n_components=4)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('heavy_components_4.csv')

'''Featurelist for ORB'''
# X = pd.read_csv('feature_list_light.csv', delimiter=None)								#All images after preprocessing								
# X = X.drop(X.columns[0], axis=1)
# X = np.array(X)
'''pca with 16 components'''
# pca = SparsePCA(n_components=16)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('light_components_16.csv')
'''pca with 8 components'''
# pca = SparsePCA(n_components=8)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('light_components_8.csv')
'''pca with 4 components'''
# pca = SparsePCA(n_components=4)
# x = pca.fit_transform(X)
# df = pd.DataFrame(x)
# df.to_csv('light_components_4.csv')


'''Hash function as mentioned in paper'''
def hamming_z(x):
	x = np.array(x)
	row,col = x.shape
	lis =[]	
	epsilon = 1e-7
	delta_kj = np.zeros((col,col))
	col_min = []
	col_max = []
	lister = []
	for j in range(col):
		for i in xrange(row):
	 		lis.append(x[i][j])
		col_min.append(np.min(np.array(lis)))
		col_max.append(np.max(np.array(lis)))
		lis = []
	for j in range(col):
		f = col_max[j]
		e =  col_min[j]
		if e - f ==0:
			for k in range(col):
				delta_kj[j][k] = 1 
		        lister.append([delta_kj[j][k],[j,k]])
		else:
			for k in range(col):
				delta_kj[j][k] = ( 1 - math.exp((-(epsilon**2)*0.5*(((k+1) * math.pi)*1.0 / (f-e))**2)))
		        lister.append([delta_kj[j][k],[j,k]])
	lister.sort()
	indexes = {}
	for l in range((col)):
		indexes[l] = lister[l]
	z = np.zeros((row,col))
	for u in range(row):
		for v in range(col):
			if (col_max[indexes[v][1][0]] - col_min[indexes[v][1][0]])==0:
				z[u][v] = 1
			else:	
				z[u][v] = math.sin(math.pi/2 + ((indexes[v][1][1]*11*x[u][indexes[v][1][0]])*1.0*math.pi/(col_max[indexes[v][1][0]] - col_min[indexes[v][1][0]])))
	
	return z




'''Binary encoding by AdaBoost thresholds'''
def thres_Z(z,labels):
	row,col = z.shape
	y = np.zeros((row,col))
	
	labels = np.array(labels)
	print row,z.shape,labels.shape
	triads = bb.triadBuilder(row,z,labels)
	t  = bb.thresholdBoost(triads,col)
	df = pd.DataFrame(t)
	df.to_csv('thresholdvalues.csv')
	for u in range(row):
		for v in range(col):
			if z[u][v] <= t[v]:
				y[u][v] = 1
			else:
				y[u][v] = -1	
	return y


'''Hamming distance between two binary encodings'''
def hamming(u, v):
    return sc.spatial.distance.hamming(u,v)


'''Distance between two encodings (Euclidean type)'''
def image_similar(y1,y2):
    return np.linalg.norm(y1-y2)

'''Used to get similarity score between all pairs of images'''
# def get_allsimilarity(y):
#     pairs = itertools.combinations(y,2)
#     similarity_score_list = []
#     for element in pairs: 
#         i = image_similar(element[0][1:],element[1][1:])
#         similarity_score_list.append([int(element[0][0]),int(element[1][0]),i])

#     return similarity_score_list


'''PCA and rest of the process for query extracted features'''
def prep(q_features,comp):
	out = []
	pca = SparsePCA(n_components=comp)
	x = pca.fit_transform(q_features)
	z = hamming_z(x)
	t = pd.read_csv('thresholdvalues.csv', delimiter=None)
	t = t.drop(t.columns[0], axis=1)
	t = t.T
	t = np.array(t)
	y = np.zeros((len(q_features),comp))

	for i in range(len(q_features)):
		for v in range(comp):
			if z[i][v] <= t[0][v]:
				y[i][v] = 1
			else:
				y[i][v] = -1	
		
	return y	


'''Read in the reduced dimensioned features'''

'''reduced dimensions after pca 1024 on CNN features'''
# X = pd.read_csv('x_features_after_pca_1024.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('z_vals_after_pca_1024.csv')

'''reduced dimensions after pca 512 on CNN features'''
# X = pd.read_csv('x_features_after_pca_512.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('z_vals_after_pca_512.csv')

'''reduced dimensions after pca 256 on CNN features'''
# X = pd.read_csv('x_features_after_pca_256.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('z_vals_after_pca_256.csv')




'''reduced dimensions after pca 16 on BRIEF features'''
# X = pd.read_csv('heavy_components_16.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_heavy_components_16.csv')


'''reduced dimensions after pca 8 on BRIEF features'''
# X = pd.read_csv('heavy_components_8.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_heavy_components_8.csv')

'''reduced dimensions after pca 4 on BRIEF features'''
# X = pd.read_csv('heavy_components_4.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_heavy_components_4.csv')

'''reduced dimensions after pca 16 on ORB features'''
# X = pd.read_csv('light_components_16.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_light_components_16.csv')


'''reduced dimensions after pca 8 on ORB features'''
# X = pd.read_csv('light_components_8.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_light_components_8.csv')

'''reduced dimensions after pca 4 on ORB features'''
# X = pd.read_csv('light_components_4.csv', delimiter = None)
# X = X.drop(X.columns[0], axis = 1)
# X = np.array(X)
# z = hamming_z(X)
# df = pd.DataFrame(z)
# df.to_csv('zvals_light_components_4.csv')




'''read in hash values for each case'''
# Z = pd.read_csv('z_vals_after_pca_1024.csv', delimiter=None)
# # Z = pd.read_csv('z_vals_after_pca_512.csv', delimiter=None)
# # Z = pd.read_csv('z_vals_after_pca_256.csv', delimiter=None)
# Z = Z.drop(Z.columns[0], axis=1)
# Z = np.array(Z)

# Z = pd.read_csv('zvals_heavy_components_16.csv', delimiter=None)
# Z = pd.read_csv('zvals_heavy_components_8.csv', delimiter=None)
# Z = pd.read_csv('zvals_heavy_components_4.csv', delimiter=None)
# Z = pd.read_csv('zvals_light_components_16.csv', delimiter=None)
# Z = pd.read_csv('zvals_light_components_8.csv', delimiter=None)
# Z = pd.read_csv('zvals_light_components_4.csv', delimiter=None)
# Z = Z.drop(Z.columns[0], axis=1)
# Z = np.array(Z)
# labels = pd.read_csv('labels.csv', delimiter=None)



'''Calculationg y-values for each case'''
# Y = thres_Z(Z,labels)
# df = pd.DataFrame(Y)
# df.to_csv('Y_after_1024.csv')
# df.to_csv('Y_after_512.csv')
# df.to_csv('Y_after_256.csv')

# df.to_csv('Y_heavy_16.csv')
# df.to_csv('Y_heavy_8.csv')
# df.to_csv('Y_heavy_4.csv')
# df.to_csv('Y_light_16.csv')
# df.to_csv('Y_light_8.csv')
# df.to_csv('Y_light_4.csv')


'''Read in the y values for each case'''
Y = pd.read_csv('Y_after_1024.csv', delimiter=None)
# Y = pd.read_csv('Y_after_512.csv', delimiter=None)
# Y = pd.read_csv('Y_after_256.csv', delimiter=None)
# Y = pd.read_csv('Y_heavy_8.csv', delimiter=None)
# Y = pd.read_csv('Y_heavy_4.csv', delimiter=None)
# Y = pd.read_csv('Y_light_16.csv', delimiter=None)
# Y = pd.read_csv('Y_light_8.csv', delimiter=None)
# Y = pd.read_csv('Y_light_4.csv', delimiter=None)

Y = Y.drop(Y.columns[0], axis=1)
Y = np.array(Y)


'''Get the query features for applying PCA and hash function'''

query_imgs = pd.read_csv('query_features_after_pca_1024.csv', delimiter=None)
# query_imgs = pd.read_csv('query_values_full_light.csv', delimiter=None)

query_imgs = query_imgs.drop(query_imgs.columns[0], axis=1)
query_imgs = np.array(query_imgs)

'''PCA and hash function on query features'''
query_imgs = prep(query_imgs,1024)
datas_query = np.array([get_image('queryfolder/'+img) for img in os.listdir('queryfolder/')])


'''Getting the  top 5 query results in a seperate folder for each query, change path accordingly '''
import os
import glob
for k in range(len(query_imgs)):
	lis_sc = []
	for i in range(Y.shape[0]):
		lis_sc.append((image_similar(query_imgs[k],Y[i]),i))
	lis_sc = sorted(lis_sc,key = lambda x: x[0])
	top_k = 5
	results = []

	mypath = '/home/safwan/Desktop/bdi/res'+str(k)
	if not os.path.isdir(mypath):
	   os.makedirs(mypath)


	for i in range(0,top_k) :
		img = np.asarray(all_images[lis_sc[i][1]])

		name = "res"+str(k)+"/result" + str(i + 1) +'.'
		if (img.ndim == 3) :
			plt.imsave(name,img,format = "png")
	plt.imsave("res"+str(k)+"/query_img"+".",datas_query[k],format = "png")


	