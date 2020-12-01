# ASDNet

This is an ongoing research project that attempts to diagnose Autism Spectrum Disorder (ASD) by classifying function magnetic resonance images (fMRIs) with convolutional neural networks (CNNs).  A great deal of preparatory work has gone into this project leaving much work still to be done in the classification/modeling area.  This project is configured for Google Colab with a Tensor Processing Unit (TPU).  

Additionally, this repo is the **1st of 3 repos** for this project, with each repo containing a logical part:  

1. Scripts/notebooks for preprocessing, modeling, storing and visualizing results.
2. Scripts/notebooks for converting the models to TensorflowLite models to be used on the Coral Edge TPU coprocessor.
3. Scripts defining the Streamlit GUI app for end-users to upload fMRIs and have the Coral Edge TPU process the inference and provide a diagnosis.

***This repo focuses on:***
**Data augmentation**
Data augmentation was explored and benchmarked and the techniques can be found in the *augmentation.Augmentor* class in the asd_tools directory.  These augmentation techniques were chosen and defined to work optimally with a TPU processor.

**Modeling architectures/hyperparameters**
