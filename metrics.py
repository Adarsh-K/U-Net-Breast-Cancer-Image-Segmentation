#### 'metrics.py' module contains the loss function metric and other metrics we use to judge the performance of our 
####	model. Metrics are supplied in 'metrics' parameter when a model is compiled
#### 	See for more detail: https://keras.io/metrics/ 

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
	"""Computes Dice coefficient for y_true, y_pred

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1)
        smooth (float): To avoid division by 0 
        See this stackoverflow discussion for explaination on y_true/y_pred: https://stackoverflow.com/a/46667294/11129457

    Returns:
        Computed Dice Coef (tensor): Dice coeffient computed on y_true and y_pred
	"""
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
	"""Computes Dice coefficient Loss which is used as the 'loss function' for training our network

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1)
        See this stackoverflow discussion for explaination on custom metrics: https://stackoverflow.com/a/45963039/11129457 

    Returns:
        Computed Dice Coef Loss(tensor): Dice coeffient loss computed on y_true and y_pred
	"""
	return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
	"""Computes Intersection-Over-Union which is used as a metric to judge our networks perfomance
		Check out this wonderful discussion on stats.stackexchange on F1 v/s iou: https://stats.stackexchange.com/a/276144

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1) 
        smooth (float): To avoid division by 0

    Returns:
        Computed IOU(tensor): IOU computed on y_true and y_pred
	"""
	intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
	union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
	iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
	return iou

def F1(y_true, y_pred, smooth=1):
	"""Computes F1 Score which is used as a metric to judge our networks perfomance

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1) 
        smooth (float): To avoid division by 0

    Returns:
        Computed F1(tensor): F1 computed on y_true and y_pred
	"""
	intersection = K.sum(y_true * y_pred, axis=[1,2,3])
	union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
	dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
	return dice

def recall(y_true, y_pred):
	"""Computes Recall which is used as a metric to judge our networks perfomance

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1) 

    Returns:
        Computed Recall(tensor): Recall computed on y_true and y_pred
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	"""Computes Precision which is used as a metric to judge our networks perfomance

    Args:
        y_true (tensor): True data of shape (batch, 256, 256, 1)
        y_pred (tensor): Output/Prediction of our network of shape (batch, 256, 256, 1) 

    Returns:
        Computed Precision(tensor): Precision computed on y_true and y_pred
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision
