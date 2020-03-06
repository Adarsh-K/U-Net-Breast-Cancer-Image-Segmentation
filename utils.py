#### 'utils.py' module is used to predict Segmentation mask for images in Test ie 'test2' dataset 

from keras.preprocessing.image import ImageDataGenerator

def predict(i, m):
	"""Predicts segmentation mask output for images in 'test2' directory

    Args:
        i (int): Test image i in 'test2' directory
        m (Model): Model which has been fit by 'fit_generator()' used to predict

    Returns:
        result (numpy.array): Numpy array of shape (17,255,255,1)- 17 predicted segmentation mask on a 
        						dataset, each of size (255,255,1)
	"""
	test_datagen = ImageDataGenerator(rescale=1./255) # Define what augmentation should take place, only rescaling here
	fname='test2/'+str(i) # Directory used for prediction is 'test2/i', see Main.ipynb for detailed Directory structure
	test_generator =test_datagen.flow_from_directory(fname, class_mode=None, color_mode="grayscale",
                                                   target_size=(256,256), batch_size=1)
	# Predicts images one-by-one to ensure ordering of image segmentation result in numpy array 'result' is same as
	# that in 'test2' directory. 
	#### v-IMP: If used a single directory with all images to predict, final ordering of segmentation
	#### 		masks is undeterministic as we can't control the order of scheduling of test images!
	#### This solves the problem of maintaining the data ordering
	#### See this issue: https://github.com/keras-team/keras/issues/5048
	result=m.predict_generator(test_generator,steps=1, verbose=1) 
	return result