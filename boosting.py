import numpy as np
import pandas as pd

epsilon = 1e-7
		
'''Triads of image1, image2, label'''
def triadBuilder(n,Z,labels) :
	triads = []
	for i in range(0,n - 1) :
		for j in range(i,n) :
			f = False														#f(u,v) = -1
			if (labels[i] == labels[j]) :
				f = True													#f(u,v) = 1
			triads.append((Z[i],Z[j],f))
	return triads

'''Adaboost learning thresholds'''
def thresholdBoost(triads,num_cols) :
	n = len(triads)
	thresholds = []
	for j in range(0,num_cols) :
		A = []
		Tn = 0																#Number of mismatched pairs in triads
		for i in range(0,n) :
			z_u,z_v,f = triads[i]
			if (z_u[j] > z_v[j]) :
				l1 = 1
			elif (z_u[j] < z_v[j]) :
				l1 = -1
			else :
				l1 = 0
			l2 = -l1
			A.append((z_u[j],l1,f))
			A.append((z_v[j],l2,f))
			if (f == -1) :
				Tn += 1
		sorted(A,key = lambda x: x[0])
		s_p,s_n = 0,0
		c_b = Tn
		Tj = A[0][0] - epsilon
		for k in range(0,len(A)) :
			z,l,f = A[k]
			if (f == 1) :
				s_p -= l
			elif (f == -1) :
				s_n -= l
			c = Tn - s_n + s_p
			if (c < c_b) :
				c_b = c
				Tj = z
		thresholds.append(Tj)
	return thresholds


