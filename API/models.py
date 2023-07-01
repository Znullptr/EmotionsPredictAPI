from django.db import models


# Create your models here.
class AudioRecords(models.Model):
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    audio_file = models.FileField(upload_to='API/audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio ID: {self.id}, Gender: {self.gender}"
