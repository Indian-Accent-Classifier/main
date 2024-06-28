# Accent Classifier for Indian Languages

Welcome to the Accent Classifier for Indian Languages! This application allows you to upload an audio file and predict the accent of the speaker among several Indian languages. The supported accents include Hindi, Kannada, Malayali, Telugu, and Tamil, each with both native and English-speaking variants.

## Features
1. Upload audio files in various formats (WAV, MP3, OGG).
2. Converts uploaded audio files to WAV format for processing.
3. Generates spectrograms from audio files.
4. Predicts the accent of the speaker using a pre-trained model.
5. Displays the top three predicted accents with their respective probabilities.

## Installation

1. Install the required dependencies: ```pip install -r requirements.txt```
2. Install Streamlit: ```pip install streamlit```

## Usage

1. To execute the application, run the following command: ```streamlit run Accent_Classifier.py```
2. Upload an audio file by clicking on the "Upload Audio File" button.
3. Click on the "Submit" button to start the prediction process.
4. The app will process the audio, generate a spectrogram, and display the top three predicted accents along with their probabilities.
5. The already deployed application can be accessed via: https://indian-accent-classifier.streamlit.app/
