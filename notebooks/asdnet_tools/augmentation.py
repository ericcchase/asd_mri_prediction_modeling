# requirements #
import tensorflow as tf 
import tensorflow_addons as tfa 
import numpy as np 
###
import random 



#*******************************       Augmentor Class       *******************************#
class Augmentor:
    '''A class used by a tensorflow Dataset to augment 4D MRI data inline and as fetched.'''
    def __init__(self, augweights):
        self.augweights = augweights
    
    @staticmethod
    @tf.function
    def rotate(volume, angles):
        volume = tf.squeeze(volume)
        volume = tfa.image.rotate(
            images=volume,
            angles=angles,
        )
        volume = tf.expand_dims(volume, axis=-1)
        return volume 
    
    @staticmethod
    @tf.function
    def rotate90(volume, k):
        volume = tf.image.rot90(
            image=volume, 
            k=k,
        )
        return volume 

    @staticmethod
    @tf.function 
    def cutout(volume):
        volume = tfa.image.random_cutout(
            images=volume,
            mask_size=16,
            constant_values=0
        )
        return volume 

    @staticmethod
    def warp(volume):
        flow_shape = [*volume.shape[:-1], 2] 
        init_flows = tf.constant(
            np.random.normal(size=flow_shape) * .5, 
            dtype=tf.float64
            )
        volume = tfa.image.dense_image_warp(volume, init_flows)
        return volume 

    @staticmethod
    @tf.function
    def noise(volume):
        gnoise = tf.random.normal(
            shape=tf.shape(volume), 
            mean=0.0, 
            stddev=0.01, 
            dtype=volume.dtype
            )
        volume = tf.add(volume, gnoise)
        return volume 


    # @tf.function
    def train_preprocessing(self, volume, label):
        '''Preprocessing function that augments data inline when the Dataset fetches it.  The data
            is augmented according to the augweights variable that determines augmentation choice
            and probability.

        Args:
            volume (Tensor): fMRI 4D tensor.
            label (int): Label for binary classification.

        Returns:
            tuple: Tuple of augmented volume tensor and its integer label.
        '''
        print('train_preprocessing...')
        funcs = random.choices(
                    [None, self.rotate, self.rotate90, self.warp, self.noise], 
                    weights=self.augweights, 
                    k=1,
                )

        for func in set(funcs):
            if func is None:
                print('no augmentation...')
                continue
            elif func.__name__ == 'rotate':
                print('rotate...')
                volume = func(
                    volume, 
                    tf.convert_to_tensor(
                        random.choice(np.linspace(-.05, .05, 10)), dtype=tf.float32 
                        )
                    )
            elif func.__name__ == 'rotate90':
                print('rotate90...')
                volume = func(volume, random.choice([1, 2, 3]))
            elif func.__name__ == 'blur':
                print('blur...')
                volume = func(volume, 2)
            else:
                print(func.__name__)
                volume = func(volume)

        # resize volume to original dims 
        volume = tf.image.resize_with_crop_or_pad(volume, 73, 61)

        return tf.convert_to_tensor(volume, tf.float64), label

