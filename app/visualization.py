import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import scipy
from nilearn import image as niimage
from nilearn import plotting


class Viz:
    def __init__(self):
        pass 
    @staticmethod
    def plot_confusion_matrix(classes: list, preds: np.ndarray) -> plt.Figure:
        pltctx = {
            'axes.edgecolor':'orange', 
            'xtick.color':'white', 
            'ytick.color':'white', 
            'text.color':'white',
            'figure.facecolor':'white'
            }

        # heatmap
        with plt.rc_context(pltctx):
            fig, ax = plt.subplots(1, 1, figsize=(2, .5))
            ax = sns.heatmap(
                preds.reshape(1, -1), 
                cmap='Blues', annot=True, linewidths=5, 
                xticklabels=classes,  yticklabels=[''],
                cbar=None, fmt='.2f', ax=ax, annot_kws={'fontsize':10}
                )
            # fluff
            plt.yticks(rotation=0, fontsize=10)
            plt.xticks(fontsize=10)
            plt.title(f'Prediction Probability', fontsize=10)
            
            return fig 

    @staticmethod
    def get_footer():
        logo_loaded = niimage.load_img('banner.nii')
        x = logo_loaded.get_data()
        x = scipy.ndimage.rotate(x, 90, axes=(2, 0))

        sns.set() 
        fig, ax = plt.subplots(1, 1, figsize=(6, 1))
        plotting.plot_epi(logo_loaded, display_mode='z', axes=ax, annotate=False, 
                        draw_cross=False, black_bg=False)
        
        return fig 
    
    @staticmethod
    def plot_uploaded_file(viz_option: str, img: np.ndarray):
            sns.set() 
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            if viz_option == 'Glass Brain':
                vizfcn = plotting.plot_glass_brain
            else:
                vizfcn = plotting.plot_epi
                
            vizfcn(img, display_mode='ortho', axes=ax, annotate=True, 
                draw_cross=True, black_bg=True, vmin=0)
            
            return fig 

