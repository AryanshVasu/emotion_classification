# Speech Emotion Classification with RAVDESS dataset
To use this model, clone this repository and run 
`
pip install -r requirements.txt
`
in a virtual environment. Create a folder eval_data in the root directory and insert audio samples to be tested.
Run `python3 eval.py` to run model inference on audio samples. Model predictions will be saved in out.csv file with the urls of the audio files. 

To run the notebook, download Audio_Speech_Actors_01-24.zip and Audio_Song_Actors_01-24.zip from [RAVDESS dataset](https://zenodo.org/records/1188976#.XCx-tc9KhQI) in a data folder and unzip into two folders. Folder structure looks like,
```
emotion_classification
├──data
| ├──song_actors
| └──speech_actors
├──eval_data 
├──audioClassification.ipynb
├──eval.py
├──model.pt
├──model_sad_removed.pt
└──requirments.txt
```

## Dataset
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The dataset contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. 

For training this model, audio only modality is used which contains 2452 files (1440 speech files + 1012 song files).
### Naming convection
The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). 
- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

## Features 
Signals are loaded with sample rate of 48kHz and cut off to be in the range of [1, 4] sewith conds. If the signal is shorter than 3s it is padded with zeros.

MFCC is generated for each audio to be used as feature.

![image](https://github.com/user-attachments/assets/9ecef162-fec3-4721-9d70-1e9a8c138162)

## Training 

80/20 stratified train-test split is used to get train and validation dataloaders. CNN is used for classification. 

### Model with all emotions 
Classification report
```
              precision    recall  f1-score   support

           0       0.81      0.92      0.86        38
           1       0.84      0.89      0.86        75
           2       0.77      0.71      0.74        75
           3       0.69      0.69      0.69        75
           4       0.88      0.99      0.93        75
           5       0.74      0.72      0.73        75
           6       0.83      0.77      0.80        39
           7       0.84      0.67      0.74        39

    accuracy                           0.80       491
   macro avg       0.80      0.79      0.80       491
weighted avg       0.79      0.80      0.79       491
```
Confusion matrix

![image](https://github.com/user-attachments/assets/53bdbf30-c79a-4c46-bd28-20342ec23366)

### Model trained with sad emotions droped
Audio with sad labels are droped to improve accuracy on other classes. CosineAnnealingWarmRestarts learning rate scheduler is used to avoid local minimas along with adam optimizer.

Learning curve

![image](https://github.com/user-attachments/assets/7f75374b-c992-41f7-bced-32061d618a06)

Classification report 
```
              precision    recall  f1-score   support

     neutral       0.74      0.97      0.84        38
        calm       0.96      0.87      0.91        75
       happy       0.77      0.68      0.72        75
       angry       0.87      0.88      0.87        75
     fearful       0.85      0.88      0.86        75
     disgust       0.76      0.74      0.75        39
   surprised       0.80      0.82      0.81        39

    accuracy                           0.83       416
   macro avg       0.82      0.83      0.82       416
weighted avg       0.83      0.83      0.83       416
```
Confusion matrix

![image](https://github.com/user-attachments/assets/6141985f-0a59-41b0-bf61-bda14a1bda1b)
