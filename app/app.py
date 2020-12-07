# requirements #
import streamlit as st 
from dipy.io.utils import Nifti1Image
import tensorflow as tf 
import numpy as np 
import datetime 
###
import pickle 
import os 
pj = os.path.join 
from visualization import Viz

############################             SETUP               ############################
st.set_page_config(page_title='ASDNet', page_icon ="☙", layout = 'centered', initial_sidebar_state = 'auto')
viz = Viz()

###########################           CSS INJECTION               ############################
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)   
local_css("style.css")


###############################         LAYOUT        #####################################

st.markdown(f"<h1 > ASD rs-fMRI Classification </h1>", unsafe_allow_html=True)
footer_fig = viz.get_footer()
st.pyplot(footer_fig, transparent=True)
# st.markdown('<br>', unsafe_allow_html=True)
st.markdown('''<strong>* Important! </strong> Ensure your <a href="https://coral.ai/products/accelerator"> 
            Coral Edge TPU coprocessor </a> is plugged into your usb port.  Also, read the 
            <strong> Data Details </strong> prior to uploading your NIfTY file.''', unsafe_allow_html=True)

#**********  ROWS & COLS  **********
r1c1 = st.beta_columns(1)[0]
r2c1, r2c2, r2c3 = st.beta_columns([1, 1, 1])
message_row = st.beta_columns(1)[0]
r3c1, r3c2 = st.beta_columns([2, 2])
#**********  COMPONENTS  **********
uploaded_file = r1c1.file_uploader('Upload your NiFTY file (*.nii)', key='file_uploader')
message_ph = message_row.empty()
# options #
model_details_option = r2c1.checkbox('Model Details', value=True, key='model_details_checkbox')
data_details_option = r2c2.checkbox('Data Details', value=True, key='asd_details_checkbox')
viz_option = r2c3.radio('Vizualization Preference:', ['Glass Brain', 'EPI'], index=1)

############################         FILE PROCESSING        ##################################
if uploaded_file:
    if uploaded_file.name.endswith('.nii'):
        try:
            img_bytestring = uploaded_file.read() 
            uploaded_file.seek(0)
            img = Nifti1Image.from_bytes(img_bytestring)       
            ###  data prep for interpretation  ###
            img_data = np.asanyarray(img.dataobj)
            data = img_data[np.newaxis,:,:,:,np.newaxis]
            data = data.astype(np.float32)
            # convert to batch!!!!!!!#
            data = np.repeat(data, 4, axis=0)

            #####################         TFLITE INTERPRETER         ##################################
            interpreter = tf.lite.Interpreter(
                model_path='./asdnet.tflite', 
                # experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')], # linux runtime library
                experimental_delegates=[tf.lite.experimental.load_delegate('libedgetpu.1.dylib')], 
            )
            input_details = interpreter.get_input_details() 
            output_details = interpreter.get_output_details()
            # allocate tpu tensor space  
            interpreter.allocate_tensors()
            interpreter.set_tensor(tensor_index=input_details[0]['index'], value=data)
            # ***  invoke!  *** #
            start = datetime.datetime.now() 
            message_ph.info('Invoking Coral TPU, be cool...')
            interpreter.invoke() 
            end_seconds = (datetime.datetime.now() - start).seconds
            message_ph.success(f'Prediction Achieved!  Execution time: {end_seconds} second(s)')
            # extract predictions
            output_data = interpreter.get_tensor(tensor_index=output_details[0]['index'])
            # revert from batch!!!!!!!#
            preds = output_data[0,...]
            classes = ['ASD', 'Typical']
            
            #####################              VISUALIZE              ##################################
            # heat map of prediciton props #
            heatmap_fig = viz.plot_confusion_matrix(classes, preds)
            r3c1.pyplot(heatmap_fig, transparent=True)
            # glass brain or epi visualization #
            uploaded_file_fig = viz.plot_uploaded_file(viz_option, img)
            r3c2.pyplot(uploaded_file_fig, transparent=True)
            
        except Exception as e:
            st.warning('We had issues, please check the run log...')
               
    else:
        message_ph.warning('That was fun, upload a NiFTY file this time.')
    

else: 
    # footer_fig = viz.get_footer()
    # st.pyplot(footer_fig, transparent=True)
    pass


if model_details_option:
    st.markdown(
        '''
        The base model used for these inferences is a convolutional neural network built in 
        Tensorflow 2 and trained on a TPU in Google Colab.  This model was then converted to 
        a Tensorflow Lite model.  The converted model is mapped onto the attached Coral TPU
        coprocessor, the data is fed to the TPU which applies the model to the data and 
        invokes an inference.
        '''
        ) 

if data_details_option:
    st.markdown(
        '''
        The data to be uploaded for diagnostic prediction is the <i> mean </i> of a functional magnetic 
        resonance image time series.  <i> The fMRI should be captured while the patient is resting. </i>
        The data is then normalized and reshaped to 61x72x61x1, with the color channel representing 
        the intensity value of the voxel. 
        ''', unsafe_allow_html=True 
        )
