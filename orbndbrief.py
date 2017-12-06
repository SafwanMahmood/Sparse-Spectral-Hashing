import numpy as np
import cv2

import os

'''Input image'''
def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    return img


'''ORB feature extractor'''
def orb_light(img):
	orb = cv2.ORB()
	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)
	print des.shape
	x = des
	x = np.average(x, axis=0)
	return x

'''BRIEF feature extractor'''
def brief_heavy(img):
	star = cv2.FeatureDetector_create("STAR")

	# Initiate BRIEF extractor
	brief = cv2.DescriptorExtractor_create("BRIEF")

	kp = star.detect(img,None)

	kp, des = brief.compute(img, kp)
	print des.shape
	x = des
	x = np.average(x, axis=0)
	return x


'''To extract features of dataset'''
# football = np.array([get_image('football/' + img) for img in os.listdir( 'football/')])
# lion = np.array([get_image('lion/' + img) for img in os.listdir('lion/')])
# guitar = np.array([get_image( 'guitar/' + img) for img in os.listdir('guitar/')])
# art = np.array([get_image( 'art/' + img) for img in os.listdir( 'art/')])
# buildings = np.array([get_image( 'buildings/' + img) for img in os.listdir('buildings/')])
# people = np.array([get_image( 'people/' + img) for img in os.listdir( 'people/')])
# piano = np.array([get_image( 'piano/' + img) for img in os.listdir( 'piano/')])
# statues = np.array([get_image( 'statues/' + img) for img in os.listdir('statues/')])

'''To extract features of queries'''
datas = np.array([get_image('queryfolder/'+img) for img in os.listdir('queryfolder/')])

# all_images = np.vstack([football,lion,guitar,art,buildings,people,piano,statues])
all_images = np.array(datas)


feautute_lis = []
feautute_lis1 = []

'''extract the features'''
x_error = []
for i in range(len(all_images)):
	try:
		x = orb_light(all_images[i])
		y = brief_heavy(all_images[i])
		feautute_lis.append(x)
		feautute_lis1.append(y)
	except Exception, e:
		feautute_lis.append(np.zeros(32))
		feautute_lis1.append(np.zeros(32))
		


import pandas as pd

'''Storing in features in CSV'''

df = pd.DataFrame(feautute_lis)
# df = df.drop(df.columns[0], axis=0)
'''For ORB'''
df.to_csv('query_values_full_light.csv')
df = pd.DataFrame(feautute_lis1)
# df = df.drop(df.columns[0], axis=0)
'''For BRIEF'''
df.to_csv('query_values_full_heavy.csv')