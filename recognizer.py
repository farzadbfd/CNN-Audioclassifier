import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import os

# for ploting the confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import itertools


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=12, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
    

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


class Recognizer():
    labels = ['1', '11', '13', '15', '17', '19', '3', '4', '5', '7', '8', '9']

    @classmethod
    def index_to_label(cls, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return cls.labels[index]

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None


    def load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()

    def predict(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)        
        tensor = transform(waveform)
        tensor = self.model(tensor.unsqueeze(0))

        prediction = get_likely_index(tensor)
        prediction = Recognizer.index_to_label(prediction.squeeze())

        return prediction

    def folder_predict(self, folder_path):
        actuals = []
        predictions = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            
            file_name = os.path.basename(f)     # file_name: a-(b).mp3
            actual = file_name.split("(")[1]    # actual: b).mp3
            actual = actual.split(")")[0]       # actual: b
            actual = int(actual)                # actual: int(b)

            prediction = int(self.predict(file_path))

            actuals.append(actual)
            predictions.append(prediction)

        return actuals, predictions


def print_prediction_stats(actuals, predictions):
    # checking the wrong predictions
    print(f'total predictions: {len(actuals)}')
    wrong_prediction = 0
    for i in range(len(actuals)):
        if(actuals[i] != predictions[i]):
            print(f'index: {i}, actual: {actuals[i]}, prediction: {predictions[i]}')
            wrong_prediction += 1

    print(f'wrong predictions: {wrong_prediction}')
    print(f'correct predictions: {len(actuals) - wrong_prediction}')        


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # loading the model
    MODEL_PATH = 'model.pt'
    recognizer = Recognizer(MODEL_PATH)
    recognizer.load_model()

    # prediction for one file
    AUDIO_PATH = 'test-15.mp3'
    prediction = recognizer.predict(AUDIO_PATH)
    print(f'{AUDIO_PATH} (one file) --> prediction: {prediction}')

    # prediction for audio files in a folder
    # file name formats should be: 'int-(int).mp3'
    FOLDER_PATH = 'test'
    actuals, predictions = recognizer.folder_predict(FOLDER_PATH)

    # printing wrong predictions.
    print_prediction_stats(actuals, predictions)

    # ploting the confusion matrix
    labels_to_int = [int(str) for str in Recognizer.labels]
    cm = metrics.confusion_matrix(actuals, predictions, labels=labels_to_int)
    plot_confusion_matrix(cm, classes=labels_to_int)

