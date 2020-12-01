# requirements #
import tensorflow as tf 
from tensorflow import keras 
import sklearn 
import scipy 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
###
import os 
pj = os.path.join 



#*******************************       Viz Class       *******************************#
class Viz:
    def __init__(self, results_path):
        self.results_path = results_path


    def plot_metrics(self, h:dict, name='Model'):
        histdf = pd.DataFrame(h)
        acc = histdf[['accuracy', 'loss']]
        acc.index = acc.index - .5
        histdf = pd.merge(histdf.drop(['accuracy', 'loss'], axis=1), acc, 
                        left_index=True, right_index=True, how='outer')
        histdf = histdf.interpolate().iloc[1:,:]

        sns.set(style='whitegrid')
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax = sns.lineplot(data=histdf[['val_accuracy', 'val_loss']], palette=sns.color_palette('Reds', 2), ax=ax)
        ax = sns.lineplot(data=histdf[['accuracy', 'loss']], palette=sns.color_palette('Blues', 2), ax=ax)
        
        plt.ylim(0, 1.)
        plt.yticks(np.arange(0, 1., .1))
        plt.legend(loc=(1.01, .5))
        plt.title(name, fontsize=15, pad=15)
        plt.xlabel('Epoch', labelpad=10)
        plt.ylabel('Metric', labelpad=10)
        # plt.savefig(pj(self.results_path, name + '.png'), dpi=400, bbox_inches='tight')
        plt.show() 


    def plot_confusion_matrix(self, m:keras.Model, x_test, y_test, class_ls):
        # predict
        pred_classes = np.argmax(m.predict(x_test), axis=-1)
        confmat = tf.math.confusion_matrix(y_test, pred_classes)    
        # heatmap
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax = sns.heatmap(
            confmat, cmap='Reds', annot=True, linewidths=5, 
            xticklabels=class_ls, yticklabels=class_ls, 
    #         cbar_kws=dict(shrink=.5),
            cbar=None, 
            ax=ax 
        )
        # fluff and save
        plt.xlabel('Predicted Class', labelpad=15)
        plt.ylabel('Actual Class', labelpad=15)
        plt.yticks(rotation=0)
        plt.title(f'Model {m.name}', fontsize=20, pad=15)
        # plt.savefig(pj(self.results_path, f"{m.name}_confmat.png"), dpi=250, bbox_inches='tight')
        plt.show() 



    def get_classification_report(self, m:keras.Model, x_test, y_test, class_ls):
        ''' Note!  The generator shuffle must be turned off or the classes will NOT align! '''
        # predict
        pred_classes = np.argmax(m.predict(x_test), axis=-1)
        
        cl_report_dict = sklearn.metrics.classification_report(y_test, 
                                                            pred_classes, 
                                                            output_dict=True)
        cl_report = pd.DataFrame(cl_report_dict).round(2)
        cl_report.rename({str(v):k for v,k in list(zip(range(len(class_ls)), class_ls))}, axis=1, inplace=True)
        # cl_report.to_excel(pj(self.results_path, f'classification_report_{m.name}.xlsx'), columns=cl_report.columns)
        return cl_report


    def plot_model_structure(self, m:keras.Model, save=False):
        if save:
            keras.utils.plot_model(model=m, to_file=pj(self.results_path, f'{m.name}_structure.png'), 
                                show_shapes=True, dpi=120)
            return 'Model Structure saved.'
        else:
            img = keras.utils.plot_model(model=m, show_shapes=True, dpi=120)
            return img 



    @staticmethod
    def get_total_vars(m: keras.Model):
        total_vars = 0
        for l in m.layers:
            try:
                total_vars += np.prod(l.trainable_variables[0].shape)
            except:
                continue
        return total_vars



    @staticmethod
    def show_slice(x):
        sns.set(style='whitegrid')
        x = x.squeeze() 
        x = scipy.ndimage.rotate(x, 90, axes=(2, 0))
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(x[30,...])
        ax[0].title.set_text('TOP')
        ax[1].imshow(x[:,30,:])
        ax[1].title.set_text('BACK')
        ax[2].imshow(x[...,30])
        ax[2].title.set_text('RIGHT')
        plt.show() 


