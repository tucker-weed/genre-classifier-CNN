import pickle
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random
import datetime

def extract(images, labels, dirlist, namelist, prefix, dim1, dim2):
	idx = 0
	pos = 0
	while len(dirlist) > 0:
		for filename in dirlist[0]:
			if filename.endswith(".png"): 
				image = Image.open(prefix + "/" + namelist[0] + "/" + filename)
				image = image.resize((dim1, dim2))
				image = np.array(image).astype('float32') / 255.0
				label = np.array([0, 0, 0, 0, 0]).astype('float32')
				label[pos] = 1.0
				labels[idx] = label
				images[idx] = image
			idx += 1
		pos += 1
		dirlist.pop(0)
		namelist.pop(0)




def get_data(prefix):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels and auxilliary values
	:param prefix: filepath
	:param segment: size of data chunk to read
	:param positionN: index to start reading from for NORMAL files
	:param positionP: index to start reading from for PNEUMONIA files
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes), two position values of saved progress
	of iteration through files, and a flag 'end' determining if dir end reached
	"""

	# PRE - PREPROCESSING

	tsplit = 45

	directoryLisB_TR = os.listdir(prefix + "/Bachata")[0 : -tsplit]
	directoryLisC_TR = os.listdir(prefix + "/Cumbia")[0 : -tsplit]
	directoryLisM_TR = os.listdir(prefix + "/Merengue")[0 : -tsplit]
	directoryLisS_TR = os.listdir(prefix + "/Salsa")[0 : -tsplit]
	directoryLisV_TR = os.listdir(prefix + "/Vallenato")[0 : -tsplit]

	tsplit -= 1

	directoryLisB_TE = os.listdir(prefix + "/Bachata")[-tsplit : ]
	directoryLisC_TE = os.listdir(prefix + "/Cumbia")[-tsplit : ]
	directoryLisM_TE = os.listdir(prefix + "/Merengue")[-tsplit : ]
	directoryLisS_TE = os.listdir(prefix + "/Salsa")[-tsplit : ]
	directoryLisV_TE = os.listdir(prefix + "/Vallenato")[-tsplit : ]
	
	NUM_INPUTS_TRAIN = len(directoryLisB_TR) + len(directoryLisC_TR) + len(directoryLisM_TR) + len(directoryLisS_TR) + len(directoryLisV_TR)
	NUM_INPUTS_TEST = len(directoryLisB_TE) + len(directoryLisC_TE) + len(directoryLisM_TE) + len(directoryLisS_TE) + len(directoryLisV_TE)

	dim1 = 180
	dim2 = 180
	num_channels = 4
	train_images = np.zeros((NUM_INPUTS_TRAIN, dim1, dim2, num_channels))
	train_labels = np.zeros((NUM_INPUTS_TRAIN, 5))
	test_images = np.zeros((NUM_INPUTS_TEST, dim1, dim2, num_channels))
	test_labels = np.zeros((NUM_INPUTS_TEST, 5))

	# TRAIN SPLIT

	dirlist = [directoryLisB_TR, directoryLisC_TR, directoryLisM_TR, 
	           directoryLisS_TR, directoryLisV_TR]
	namelist = ["Bachata", "Cumbia", "Merengue", "Salsa", "Vallenato"]

	extract(train_images, train_labels, dirlist, namelist, prefix, dim1, dim2)

	directoryLisB_TR = None
	directoryLisC_TR = None
	directoryLisM_TR = None
	directoryLisS_TR = None
	directoryLisV_TR = None

	# TEST SPLIT

	dirlist = [directoryLisB_TE, directoryLisC_TE, directoryLisM_TE, 
	           directoryLisS_TE, directoryLisV_TE]
	namelist = ["Bachata", "Cumbia", "Merengue", "Salsa", "Vallenato"]

	extract(test_images, test_labels, dirlist, namelist, prefix, dim1, dim2)


	return train_images, train_labels, test_images, test_labels
