#### 'plots.py' module is used to: 
####	1. Plot "training curve" of the network/model(for various metrics as given in 'titles' below)  
####	2. Show for images- model's segmentation prediction, obtained "Binary mask" and "Ground Truth" segmentation

import matplotlib.pyplot as plt, random, numpy as np, cv2
from PIL import Image
from keras import backend as K

def training_history_plot(results):
	"""Plots "training curve" for the network/model for metrics listed below:
    		1. Dice loss
    		2. Pixel-wise accuracy 
    		3. Intersection Over Union(IOU)
    		4. F1 score
    		5. Recall
    		6. Precision

    Args:
        results (History): Output of 'model.fit_generator()', 'History.history' attribute is a record of metrics
        					values as described above(from 1-6)

    Returns:
        None
	"""
	titles = ['Dice Loss','Accuracy','IOU','F1','Recall','Precision'] 
	metric = ['loss', 'acc', 'iou','F1','recall','precision'] # Metrics we're keeping track off

	# Define specification of our plot
	fig, axs = plt.subplots(3,2, figsize=(15, 15), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.5, wspace=0.2)
	axs = axs.ravel()

	for i in range(6):
		axs[i].plot(results.history[metric[i]]) # Calls from 'History.history'- 'metric[i]', note 'results' is 
		axs[i].set_title(titles[i])				# a 'History' object
		axs[i].set_xlabel('epoch')  
		axs[i].set_ylabel(metric[i])
		axs[i].legend(['train'], loc='upper left')


def model_prediction_plot(results, t=0.2):
	"""Displays:
    		1. Original test image  
    		2. Network's predicted segmentation mask 
    		3. Binary mask obtained from 2
    		4. Ground truth segmentation for the test image

    Args:
        results (numpy.array): Numpy array of shape (17,255,255,1)- 17 predicted segmentation mask, each of size
        						(255,255,1)
        t (float)(Default=0.2): Threshold used to convert predicted mask to binary mask

    Returns:
        None
	"""
	bin_result = (results >= t) * 1 # Convert predicted segmentation mask to binary mask on threshold 't'
	titles=['Image','Predicted Mask','Binary Mask','Ground Truth']
	r=random.sample(range(17),4) # Random sample for test images to display

	# Define specification of our plot
	fig, axs = plt.subplots(4, 4, figsize=(15, 15), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.5, wspace=0.2)
	axs = axs.ravel()

	for i in range(4): # 1 iteration for each selected test image
		# Displays test image 
		axs[(i*4)+0].set_title(titles[0])
		fname = 'test/images/img/'+str(r[i])+'.png'
		image = Image.open(fname).convert("L")
		arr = np.asarray(image)
		axs[(i*4)+0].imshow(arr/255, cmap='gray')

		# Displays predicted segmentation mask
		axs[(i*4)+1].set_title(titles[1])
		I=np.squeeze(results[r[i],:,:,:])
		axs[(i*4)+1].imshow(I, cmap="gray")

		# Displays binary mask
		axs[(i*4)+2].set_title(titles[2])
		I=np.squeeze(bin_result[r[i],:,:,:])
		axs[(i*4)+2].imshow(I, cmap="gray")

		# Displays Ground truth segmentation mask 
		axs[(i*4)+3].set_title(titles[3])
		fname = 'test/label/img/'+str(r[i])+'.png'
		image = Image.open(fname).convert("L")
		arr = np.asarray(image)
		axs[(i*4)+3].imshow(arr/255, cmap='gray')

def canny_compare_plot(results, results_canny):
	"""Compares model's performance on the "standard" dataset and dataset "overlayed" with "canny edges"
    	Displays:
    		1. Original test image  
    		2. Predicted segmentation mask on "standard" dataset
    		3. Predicted segmentation mask on "overlayed" dataset
			4. Binary mask for "standard" dataset
			5. Binary mask for "overlayed" dataset
    		6. Ground truth segmentation for the test image

    Args:
        results (numpy.array): Numpy array of shape (17,255,255,1)- 17 predicted segmentation mask on "standard" 
        						dataset, each of size (255,255,1)
        results_canny (numpy.array): Numpy array of shape (17,255,255,1)- 17 predicted segmentation mask on 
        						"overlayed", each of size (255,255,1)

    Returns:
        None
	"""
	bin_result = (results >= 0.1) * 1 # Convert "standard" predicted segmentation mask to binary mask 
	bin_result_canny = (results_canny >= 0.2) * 1 # Convert "overlayed" predicted segmentation mask to binary mask 
	titles=['Image','Predicted Mask','Predicted Mask Canny','Binary Mask','Binary Mask Canny','Ground Truth']
	r=random.sample(range(17),4) # Random sample for test images to display

	# Define specification of our plot 
	fig, axs = plt.subplots(4, 6, figsize=(15, 15), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.5, wspace=0.2)
	axs = axs.ravel()

	for i in range(4): # 1 iteration for each selected test image
		# Displays test image 
		axs[(i*6)+0].set_title(titles[0])
		fname = 'test/images/img/'+str(r[i])+'.png'
		image = Image.open(fname).convert("L")
		arr = np.asarray(image)
		axs[(i*6)+0].imshow(arr/255, cmap='gray')

		# Displays "standard" predicted segmentation mask
		axs[(i*6)+1].set_title(titles[1])
		I=np.squeeze(results[r[i],:,:,:])
		axs[(i*6)+1].imshow(I, cmap="gray")

		# Displays "overlayed" predicted segmentation mask
		axs[(i*6)+2].set_title(titles[3])
		I=np.squeeze(results_canny[r[i],:,:,:])
		axs[(i*6)+2].imshow(I, cmap="gray")

		# Displays "standard" binary mask
		axs[(i*6)+3].set_title(titles[2])
		I=np.squeeze(bin_result[r[i],:,:,:])
		axs[(i*6)+3].imshow(I, cmap="gray")

		# Displays "overlayed" binary mask
		axs[(i*6)+4].set_title(titles[4])
		I=np.squeeze(bin_result_canny[r[i],:,:,:])
		axs[(i*6)+4].imshow(I, cmap="gray")

		# Displays Ground truth segmentation mask 
		axs[(i*6)+5].set_title(titles[5])
		fname = 'test/label/img/'+str(r[i])+'.png'
		image = Image.open(fname).convert("L")
		arr = np.asarray(image)
		axs[(i*6)+5].imshow(arr/255, cmap='gray')

def activation_map(image, layer, channel, m_c):
	"""Displays:
    		1. Original test image  
    		2. Activation Map for provided layer and channel
    		3. Transparent overlay of Activation Map over test image

    Args:
        image (file name): Location of test image
        layer (int): Layer number, can be found from model summary
        channel (int): Channel number in the 'layer', number of channels in provided layer can be found from 
        		model summary
        m_c (Model): Keras Model object used as network

    Returns:
        None
	"""

	# Define specification of our plot 
	fig, axs = plt.subplots(1, 3, figsize=(20, 20), facecolor='w', edgecolor='k')
	fig.subplots_adjust(wspace=0.2)
	axs = axs.ravel()

	# Displays Original test image 
	ori=cv2.imread(image)
	axs[0].set_title('Original Image')
	axs[0].imshow(ori)
	
	# Displays Activation Map of test image for 'layer' and 'channel' 
	img=cv2.imread(image, 0) # Reading image as "Grayscale" to ensure it's dimensions 1D ie. same as the Activation
							# Map, also images feed into the network are "Grayscale"
	x=cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
	x=np.expand_dims(x, axis=2) # Making images 4D as input should be 4D
	x=np.expand_dims(x, axis=0)

	# Keras function that will return the output of 'layer' given a input test image
	get_layer_output = K.function([m_c.layers[0].input, K.learning_phase()],
                                  [m_c.layers[layer].output])
	layer_output = get_layer_output([x, 0])[0] 
	act=layer_output[0, :, :,channel] # Visualising 'channel' of 'layer'
	act=cv2.resize(act, (512,512))
	act=act/255

	axs[1].set_title('Activation Map')
	axs[1].imshow(act,cmap='jet')

	# Displays Activation Map "overlayyed" on test image
	img=img.astype('float32')
	img=cv2.resize(img, (512,512))
	img=img/255
	dst=cv2.addWeighted(img,0.5,act,0.5,0) # Giving equal weightage to both images
	axs[2].set_title('Overlayed')
	axs[2].imshow(dst, cmap='jet')