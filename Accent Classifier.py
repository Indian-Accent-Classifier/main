import streamlit as st
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import keras 
import tensorflow as tf
import time

st.set_page_config(
    page_title="Accent Predictorr",
    page_icon="images/icon.png",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Indian Accent Classifier"
    }
)

st.sidebar.image('images/icon.png')

st.markdown(
    """
    <style>
    [data-testid="stFileUploadDropzone"] {
        margin-top: -30px;
    }
    """,
    unsafe_allow_html=True,
)



st.write("## Accent Classifier for Indian Languages")

st.write("## Upload Audio File")
audio = st.file_uploader("", type=["wav"])

# if audio is not None:
#     # audio_path = os.path.join('', "file.wav")
#     with open(audio, "wb") as f:
#         f.write(audio.getbuffer())
    #st.success("File saved successfully!")

# loaded_model = pickle.load('model.pkl')

# model = models.Sequential()
loaded_model = keras.models.load_model("/content/drive/MyDrive/model.h5")



if st.button("Submit"):
    def generate_spectrogram(output_dir):
        y, sr = librosa.load(audio, sr=22050) # load audio file
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000) # generate spectrogram
        S_dB = librosa.power_to_db(S, ref=np.max) # convert to dB scale
        plt.figure(figsize=(2.56, 2.56), dpi=100) # set image size
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', fmax=8000) # plot spectrogram
        plt.axis('off') # remove axis labels
        plt.savefig(os.path.join(output_dir, os.path.splitext(os.path.basename("file"))[0] + '.png'), bbox_inches='tight', pad_inches=0) # save image
        plt.close() # close plot
    
    generate_spectrogram('')
    class_names = ['Hi_En','Hi_Hi','Ka_En','Ka_Ka','Ma_En','Ma_Ma','Ta_Ta','Te_En','Te_Tes']
    img = tf.keras.preprocessing.image.load_img('file.png', target_size=(128, 431))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0 # convert image to array and normalize
    img_array = np.expand_dims(img_array, axis=0) # add batch dimension
    probabilities = loaded_model.predict(img_array)[0] # predicst probabilities for each class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx] # extract predicted class index
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probabilities[i]}")
    print(predicted_class)
    st.write("Prdecting your accent... This may take a while...")
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.write("## Predicted Class: ", predicted_class)