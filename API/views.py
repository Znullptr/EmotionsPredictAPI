import os
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .models import AudioRecords
from .serializers import AudioSerializers
import librosa
import joblib
import numpy as np
import pandas as pd


def extract_mfcc_features(file_path):
    try:
        audio, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr)
        return mfcc.T
    except Exception:
        error_message = 'Unsupported audio format. Please provide a valid audio file such WAV,MP3,AAC'
        os.remove(file_path)
        raise ValueError(error_message)


def predict_emotion(mfcc_features, gender):
    mdl = joblib.load('API/emotion_predict_model.pkl')
    scaler = joblib.load('API/scaler.pkl')
    lb = joblib.load('API/emotion_label_encoder.pkl')
    gender = gender.replace('M', '1').replace('F', '0')
    mfcc_stats = np.hstack(
        (np.mean(mfcc_features, axis=0), np.std(mfcc_features, axis=0), np.max(mfcc_features, axis=0)))
    # Create a dictionary with the aggregated statistics
    mfcc_dict = {f'mfcc{j}_mean': mfcc_stats[j] for j in range(mfcc_stats.shape[0] // 3)}
    mfcc_dict.update(
        {f'mfcc{j}_std': mfcc_stats[j + mfcc_stats.shape[0] // 3] for j in range(mfcc_stats.shape[0] // 3)})
    mfcc_dict.update(
        {f'mfcc{j}_max': mfcc_stats[j + 2 * mfcc_stats.shape[0] // 3] for j in range(mfcc_stats.shape[0] // 3)})
    df_processed = pd.DataFrame.from_records([mfcc_dict])
    df_processed.insert(0, 'gender', gender)

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(df_processed)

    # Predict emotion using the model
    prediction = mdl.predict(scaled_data)
    predicted_emotion = lb.inverse_transform(prediction)
    # Assuming the output of the model is one-hot encoded, find the emotion with the highest probability
    # and convert it back to the original label
    return predicted_emotion


def perform_prediction(validated_data):
    file_path = 'API/audio_files/' + validated_data['audio_file'].name
    gender = validated_data['gender']
    mfcc_features = extract_mfcc_features(file_path)
    emotion = predict_emotion(mfcc_features, gender)
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


def welcome(request):
    api_url = "/api/"

    context = {
        'api_url': api_url
    }

    return render(request, 'welcome.html', context)
