from rest_framework import serializers
from .models import AudioRecords


class AudioSerializers(serializers.ModelSerializer):
    class Meta:
        model = AudioRecords
        fields = '__all__'
