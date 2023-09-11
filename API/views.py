import os
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .models import AudioRecords
from .serializers import AudioSerializers
import librosa
import joblib
import numpy as np
from keras.models import load_model


# Function to extract audio features
def extract_features(data, sr, hop_length=512, win_length=2048):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, hop_length=hop_length, win_length=win_length)
    chroma = librosa.feature.chroma_stft(y=data, sr=sr, hop_length=hop_length, win_length=win_length)
    sc = librosa.feature.spectral_contrast(y=data, sr=sr, hop_length=hop_length, win_length=win_length)
    zre = librosa.feature.zero_crossing_rate(y=data, hop_length=hop_length, frame_length=win_length)
    rms = librosa.feature.rms(y=data, hop_length=hop_length, frame_length=win_length)
    result = np.vstack((mfcc, chroma, sc, zre, rms))
    return result.T


def to_statistical(feature):
    # Compute statistics (mean, standard deviation, maximum and minimum) along the frames
    mfcc_stats = np.hstack(
        (np.mean(feature, axis=0), np.std(feature, axis=0), np.max(feature, axis=0), np.min(feature, axis=0)))
    return mfcc_stats


def get_features(path, duration=2.5, offset=0.6):
    try:
        data, sr = librosa.load(path, duration=duration, offset=offset)
        aud = extract_features(data, sr)
        audio = [np.array(to_statistical(aud))]
        return audio
    except Exception:
        error_message = 'Unsupported audio format. Please provide a valid audio file such WAV,MP3,AAC'
        os.remove(path)
        raise ValueError(error_message)


def predict_emotion(path):
    mdl = load_model('API/model')
    scaler = joblib.load('API/scaler.pkl')
    lb = joblib.load('API/label_encoder.pkl')
    features = get_features(path)
    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(features)

    # Predict emotion using the model
    predictions = mdl.predict(scaled_data)
    y_pred = predictions.argmax(axis=1)
    prediction = y_pred.astype(int).flatten()
    predicted_emotion = lb.inverse_transform(prediction)
    # Assuming the output of the model is one-hot encoded, find the emotion with the highest probability
    # and convert it back to the original label
    return predicted_emotion


def perform_prediction(validated_data):
    file_path = 'API/audio_files/' + validated_data['audio_file'].name
    emotion = predict_emotion(file_path)
    os.remove(file_path)
    return emotion


class AudiosView(viewsets.ModelViewSet):
    queryset = AudioRecords.objects.all()
    serializer_class = AudioSerializers

    def create(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            prediction_result = perform_prediction(serializer.validated_data)
            response_data = {'Emotion_predicted': prediction_result}
            headers = self.get_success_headers(serializer.data)
            return Response(response_data, status=201, headers=headers)
        except ValueError as e:
            error_message = str(e)
            return Response({'message': error_message}, status=400)

