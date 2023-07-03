# Speech Emotions Prediction API
Tahar Jaafer

## Project Overview

Through all the available senses humans can actually sense the emotional state of their communication partner. The emotional detection is natural for humans but it is very difficult task for computers; although they can easily understand content based information, accessing the depth behind content is difficult and that’s what speech emotion recognition (SER) sets out to do. It is a system through which various audio speech files are classified into different emotions such as happy, sad, anger and neutral by computer. SER can be used in areas such as the medical field or customer call centers. With this project I hope to look into applying this model into an app that individuals with ASD can use when speaking to others to help guide conversation and create/maintain healthy relationships with others who have deficits in understanding others emotions.
## Dataset  

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) Dataset from Kaggle contains 1440 audio files from 24 Actors vocalizing two lexically-matched statements. Emotions include angry, happy, sad, fearful, calm, neutral, disgust, and surprised. [Click for dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)


## Process  

See **Saved_Models/prediction_model.ipynb**  

1)	Loaded audio files, created visualizations, conducted feature extraction (mfcc features extraction and used statistics values to describe the mfcc matrix like mean , std and max ) resulting into dataframe (see **Saved_Models/audio.csv**)
   
2)	Built SVM Model with cross validation with 6 V-folds and grid search to tune the hyperparameters of the model,  Obtained an accuracy score of 80%.
   - *SVM Model*
     
<p align="center">
  <img width="600" height="500" src="https://github.com/Znullptr/EmotionsPredictAPI/blob/main/Images_Uploads/SVM_Confusion_Matrix.png">
</p>  

3)	Implemented inital 1D CNN Model. Obtained an accuracy score of 61% with the model having difficulty classifying angry, fear , sad and disgust.
   - *CNN Model*

<p align="center">
  <img width="600" height="500" src="https://github.com/Znullptr/EmotionsPredictAPI/blob/main/Images_Uploads/CNN_Confusion_Matrix.png">
</p>  

4)  Plot the learning curve for both the CNN and SVM models and the validation score was significatlly better in SVM while the CNN was not learning as 
    well as SVM with the same training examples used to feed them and has caused an overfitting of data.

<p align="center">
  <img width="800" height="500" src="https://github.com/Znullptr/EmotionsPredictAPI/blob/main/Images_Uploads/Learning_Curve_CNN_VS_SVM.png">
</p>  
    
5)  I decided then to pickle the SVM model because of its higher accuracy and better learning curve to use it in my api.
   
7)	See **Test_Uploads** for all my test sample audio files i used to test.

## Limitations  

Limitations include not using feature selection to reduce the dimensionality of my augmented CNN which may have improved learning performance and i didn't use cross-validation to tune my hyperparameters like i did with SVM. Another limitation included using minimal data, the RAVDESS Dataset has only 1,440 files which may be why there was overfitting of the data. Additional datasets could have been utilized.

## Next Steps

Next steps for this project,  

i built a Django REST API to detect emotion and was deployed on render under this URL:https://emotionpredict.onrender.com .    

Afterwards, I would like to use this api to build system that can recognize emotion in real time and then calculate degree of affection such as love, truthfulness, and friendship of the person you are talking to.

# Technologies Used  

Django: Python web framework for building the API.  

Django REST Framework: Extension for building RESTful APIs with Django.  

Tensorflow && sikit-learn: Machine learning library for training the loan approval prediction model.  

Render: Cloud platform for deploying the API. 

# License  

This project is licensed under the MIT License.

