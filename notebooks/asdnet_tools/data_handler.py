# requirements #
import numpy as np 
import nilearn.image as niimage 
import tensorflow as tf 
from tensorflow import keras 
from google.cloud import storage
###
import os 
pj = os.path.join 
import pickle 




#*******************************       DataHandler Class       *******************************#
class DataHandler:
    '''A convenience class to fetch data from a GCS bucket, fetch a Tensorflow Dataset of the data, 
       or to save the model back to GCS.  Requires a service account JSON file to be accessible to 
       the instance.'''
    
    def __init__(self, service_acct_fpath, bucket_name):
        '''
        Args:
            service_acct_fpath (str): File path to the GCS service account credential JSON file.
            bucket_name (str): Name of the bucket with either NIfTY data or pickled data.
        '''
        self.gcs = storage.Client.from_service_account_json(service_acct_fpath)
        self.bucket = self.gcs.get_bucket(bucket_name)
        
    def get_nifty_data_arrays(self, site_ls=None, max_results=None):
        if site_ls is None:
            site_ls = [
                'CMUa', 'CMUb', 'Caltech', 'KKI', 'Leuven1', 'Leuven2', 
                'MaxMuna', 'MaxMunb', 'MaxMunc', 'MaxMund', 'NYU', 'OHSU', 
                'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford', 'Trinity', 
                'UCLA1', 'UCLA2', 'UM1', 'UM2', 'USM', 'Yale'
                ]
        X, y = [], []
        for site in site_ls:
            for label in [0, 1]:
                for blob in self.bucket.list_blobs(max_results=max_results, prefix=f'{site}/{label}'):
                    if not blob.name.endswith('.nii'):
                        print('bad file, skipping ', blob.name)
                        continue
                    blob.download_to_file(open('tempfile.nii', 'wb'))
                    nii_img_obj = niimage.load_img('tempfile.nii')
                    img_data = nii_img_obj.get_fdata() 
                    img_data = np.expand_dims(img_data, axis=-1)
                    X.append(img_data)
                    y.append(label)
        X = np.asanyarray(X, dtype=np.float64)
        y = np.asanyarray(y, dtype=np.int64)
        return X, y 


    def get_pickled_data_arrays(self, site_ls=None, max_results=None):
        if site_ls is None:
            site_ls = [
                'CMUa', 'CMUb', 'Caltech', 'KKI', 'Leuven1', 'Leuven2', 
                'MaxMuna', 'MaxMunb', 'MaxMunc', 'MaxMund', 'NYU', 'OHSU', 
                'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford', 'Trinity', 
                'UCLA1', 'UCLA2', 'UM1', 'UM2', 'USM', 'Yale'
                ]
        X, y = [], []
        for site in site_ls:
            for label in [0, 1]:
                for blob in self.bucket.list_blobs(max_results=max_results, prefix=f'{site}/{label}'):
                    if not blob.name.endswith('.pkl'):
                        print('bad file, skipping ', blob.name)
                        continue
                    blob.download_to_file(open('tempfile.pkl', 'wb'))
                    img_data = pickle.load(open('tempfile.pkl', 'rb'))
                    X.append(img_data)
                    y.append(label)
        X = np.asanyarray(X, dtype=np.float64)
        y = np.asanyarray(y, dtype=np.int64)
        return X, y 


    def save_model_gcp(self, m: keras.Model, bucket_name, init=False):
        os.environ['MODEL_NAME'] = m.name 
        bucket = self.gcs.get_bucket(bucket_name)
        try:
            if init:
                # save initial weights #
                m.save_weights(f'/content/{m.name}_init_weights.h5', overwrite=True, save_format='h5')
                blob = bucket.blob(f'results/{m.name}_init_weights.h5')
                blob.upload_from_filename(f'/content/{m.name}_init_weights.h5')
            else:
                # save trained model #
                m.save(f'/content/{m.name}_trained', overwrite=True)
                # blob = bucket.blob(f'/content/{m.name}_trained')
                # save weights #
                m.save_weights(f'/content/{m.name}_trained_weights.h5', overwrite=True, save_format='h5')
                blob = bucket.blob(f'results/{m.name}_trained_weights.h5')
                blob.upload_from_filename(f'/content/{m.name}_trained_weights.h5') 
        except Exception as e:
            print('Issues saving to GCP...')
            print(e)
            return False
        return True
    
    
    def get_dataset(self, x, y, batch_size, preprocessing_fcn=None, train=False) -> tf.data.Dataset:
        if train:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            AUTO = tf.data.experimental.AUTOTUNE # decouple the time when data is produced from the time when data is consumed
            dataset = dataset.shuffle(x.shape[0]) \
                            .cache() \
                            .repeat() \
                            .map(preprocessing_fcn) \
                            .batch(batch_size, drop_remainder=True) \
                            .prefetch(AUTO) 
            return dataset
        else:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            AUTO = tf.data.experimental.AUTOTUNE # decouple the time when data is produced from the time when data is consumed
            dataset = dataset.shuffle(x.shape[0]) \
                            .cache() \
                            .batch(batch_size, drop_remainder=False) \
                            .prefetch(AUTO) 
            return dataset
        