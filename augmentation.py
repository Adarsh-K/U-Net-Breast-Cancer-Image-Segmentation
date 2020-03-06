#### 'augmentation.py' module is used to perform Data Augmentation and/or return a Generator used in 
#### 'model.fit_generator()' and 'model.evaluate_generator()'

from keras.preprocessing.image import ImageDataGenerator

def train_data_aug(canny=False):
    """Peforms real-time Data Augmentation on the Training dataset used in 'model.fit_generator()'

    Args:
        canny (bool): If True performs augmentation on 'train_canny' directory else on 'train' directory

    Returns:
        iterator: Single generator "zipped" to maintain mapping of train image and corresponding mask 
            after augmentation
    """    
    seed = 1 # To ensure correct mapping of train images to corresponding mask, otherwise ordering  
             # of images and masks aren't consistent 

    # Define what augmentation should take place
    image_datagen = ImageDataGenerator(rotation_range=0.2, rescale=1./255, width_shift_range=0.05, 
                    height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                    horizontal_flip=True, fill_mode='nearest') 
    mask_datagen = ImageDataGenerator(rotation_range=0.2, rescale=1./255, width_shift_range=0.05,
                    height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                    horizontal_flip=True, fill_mode='nearest')  
    dir='train/' # Directory containing train image dataset
    if(canny):
        dir='train_canny/' # Directory containing train images overlayed with their "Canny edges" 
    image_generator =image_datagen.flow_from_directory(dir+'images', class_mode=None, seed=seed, 
                    color_mode="grayscale", target_size=(256,256), batch_size=2)
    mask_generator = mask_datagen.flow_from_directory(dir+'label', class_mode=None, seed=seed, 
                    color_mode="grayscale", target_size=(256,256), batch_size=2)

    train_generator = zip(image_generator, mask_generator) # Zipped to ensure correct ordering/mapping of images
                                                           # and corresponding masks
    return train_generator

def test_data_aug(canny=False):
    """Peforms real-time Data Augmentation on the Test/Validation dataset used in 'model.evaluate_generator()'

    Args:
        canny (bool): If True performs augmentation on 'test_canny' directory else on 'test' directory

    Returns:
        iterator: Single generator "zipped" to maintain mapping of test image and corresponding mask 
            after augmentation
    """ 
    seed = 1 # To ensure correct mapping, same as above
    image_datagen1 = ImageDataGenerator(rescale=1./255) # Required only rescaling as we're testing here thus 
                                                        # no augmentation
    mask_datagen1 = ImageDataGenerator(rescale=1./255)  
    dir='test/' # Directory containing test image dataset
    if(canny):   
        dir='test_canny/' # Directory containing test images overlayed with their "Canny edges"
    image_generator1 =image_datagen1.flow_from_directory(dir+'images', shuffle=False, class_mode=None, 
                    seed=seed, color_mode="grayscale", target_size=(256,256), batch_size=1)
    mask_generator1 = mask_datagen1.flow_from_directory(dir+'label', shuffle=False, class_mode=None, 
                    seed=seed, color_mode="grayscale", target_size=(256,256), batch_size=1)

    test_generator = zip(image_generator1, mask_generator1) # Zipped to ensure correct ordering/mapping of images
                                                            # and corresponding masks
    return test_generator