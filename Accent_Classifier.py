import streamlit as st
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import keras
from pydub import AudioSegment
from collections import Counter

st.set_page_config(
    page_title="Accent Predictor",
    page_icon="images/icon.png",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# Indian Accent Classifier"
    }
)

st.sidebar.image("images/icon.png")

st.markdown(
    """
    <style>
    [data-testid="stFileUploadDropzone"] {
        margin-top: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("## Accent Classifier for Indian Languages")

st.write("## Upload Audio File")
audio = st.file_uploader("", type=["wav", "mp3", "ogg"])

loaded_model = keras.models.load_model("model.h5")

if st.button("Submit"):
    def convert_to_wav(file):
        audio_segment = AudioSegment.from_file(file)
        wav_file = "converted.wav"
        audio_segment.export(wav_file, format="wav")
        return wav_file

    def generate_spectrogram(output_dir):
        y, sr = librosa.load(audio_file, sr=22050)  # load audio file
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # generate spectrogram
        S_dB = librosa.power_to_db(S, ref=np.max)  # convert to dB scale
        plt.figure(figsize=(2.56, 2.56), dpi=100)  # set image size
        librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", fmax=8000)  # plot spectrogram
        plt.axis("off")  # remove axis labels
        plt.savefig(
            os.path.join(output_dir, os.path.splitext(os.path.basename("file"))[0] + ".png"),
            bbox_inches="tight", pad_inches=0)  # save image
        plt.close()  # close plot

    if audio:
        if audio.name.endswith(".wav"):
            audio_file = audio
        else:
            audio_file = convert_to_wav(audio)
        
        generate_spectrogram("")
        class_names = ["Hi_En", "Hi_Hi", "Ka_En", "Ka_Ka", "Ma_En", "Ma_Ma", "Ta_Ta", "Te_En", "Te_Te"]
        img = tf.keras.preprocessing.image.load_img("file.png", target_size=(128, 431))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # convert image to array and normalize
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        probabilities = loaded_model.predict(img_array)[0]  # predict probabilities for each class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]  # extract predicted class index
        d = {}
        for i in range(len(class_names)):
            print(f"{class_names[i]}: {probabilities[i]}")
            d[class_names[i]] = probabilities[i]
        
        st.write("Predicting your accent... This may take a while...")
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        if(d["Hi_En"] > d["Hi_Hi"]):
            del d["Hi_Hi"]
        else:
            del d["Hi_En"]

        if(d["Ka_En"] > d["Ka_Ka"]):
            del d["Ka_Ka"]
        else:
            del d["Ka_En"]

        if(d["Te_En"] > d["Te_Te"]):
            del d["Te_Te"]
        else:
            del d["Te_En"]

        if(d["Ma_En"] > d["Ma_Ma"]):
            del d["Ma_Ma"]
        else:
            del d["Ma_En"]

        high = Counter(d).most_common(3)
        
        st.write("## You have the below Accents")

        for i in high:

            if(i[0] == "Hi_En" or i[0] == "Hi_Hi"):
                st.write("### Hindi Accent : ",round(i[1]*100,3),"%")
            elif(i[0] == "Ka_En" or i[0] == "Ka_Ka"):
                st.write("### Kannada Accent : ",round(i[1]*100,3),"%")
            elif(i[0] == "Ma_En" or i[0] == "Ma_Ma"):
                st.write("### Malayali Accent : ",round(i[1]*100,3),"%")
            elif(i[0] == "Te_En" or i[0] == "Te_Te"):
                st.write("### Telugu Accent : ",round(i[1]*100,3),"%")
            else:
                st.write("### Tamil Accent : ",round(i[1]*100,3),"%")
    else:
        st.warning("Upload the Audio file first!")
