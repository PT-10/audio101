import torch
import librosa
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from sklearn.model_selection import train_test_split


# Custom Dataset Class for UrbanSound8K
class UrbanSound8KDataset(Dataset):
    def __init__(self, audio_dir, file='UrbanSound8K/metadata/UrbanSound8K.csv', sample_rate=22050, transform=None):
        """
        Args:
            audio_dir (str): Path to the directory containing audio files.
            file (str): Path to the CSV file containing metadata (with fsID, classID, etc.).
            sample_rate (int): The sample rate to resample audio to (default is 22050).
            transform (callable, optional): A function/transform to apply to the audio (e.g., MFCC).
        """
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
        self.sample_rate = sample_rate
        self.transform = transform
        
        # Label encoding for classID to numerical labels
        self.label_encoder = LabelEncoder()
        self.metadata['classID'] = self.label_encoder.fit_transform(self.metadata['class'])

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.metadata)

    def extract_features(self, X):
        result = np.array([])

        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=self.sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        

        # Chroma_STFT
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sample_rate, n_chroma=32, window="hamming", n_fft=1024).T, axis=0)
        result = np.hstack((result, chroma))

        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=self.sample_rate, n_mels=128, fmax=8000, window="hamming", n_fft=1024, hop_length=512).T, axis=0)
        result = np.hstack((result, mel))
        print(mel.shape)

        # Zero Crossing Rate
        Z = np.mean(librosa.feature.zero_crossing_rate(y=X), axis=1)
        result = np.hstack((result, Z))
        print(Z.shape)

        # Root Mean Square Energy
        rms = np.mean(librosa.feature.rms(y=X).T, axis=0)
        result = np.hstack((result, rms))

        return result

    def __getitem__(self, idx):
        """Return the sample (audio, label, metadata) at index `idx`."""
        # Get the metadata for the current sample
        row = self.metadata.iloc[idx]
        start_time = row['start']
        end_time = row['end']
        fold = row['fold']
        file_name = row['slice_file_name']
        label = row['classID']
        
        # Load the audio file using librosa
        audio_path = os.path.join(self.audio_dir, f"fold{fold}", file_name)
        waveform, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        
        # Resample if the sample rate does not match the desired rate
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(waveform, sample_rate, self.sample_rate)
        
        # Extract features
        features = self.extract_features(waveform)
        
        # Convert features to tensor and label to long
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)  # Ensure label is of type long

        sample = {
            'features': features_tensor,
            'start': start_time,
            'end': end_time,
            'fold': fold,
            'file_name': file_name,
            'label': label_tensor
        }
        
        return sample



class AudioClassifierANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassifierANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # Raw logits, no softmax here

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits
        return x



# Create function to split dataset into train, validation, and test sets
def create_datasets(audio_dir, file, test_size=0.2, val_size=0.2):
    dataset = UrbanSound8KDataset(audio_dir=audio_dir, file=file)
    
    # Splitting the data into train, validation, and test sets
    train_metadata, temp_metadata = train_test_split(dataset.metadata, test_size=test_size + val_size, stratify=dataset.metadata['classID'])
    val_metadata, test_metadata = train_test_split(temp_metadata, test_size=test_size / (test_size + val_size), stratify=temp_metadata['classID'])
    
    # Create Dataset instances for train, validation, and test
    train_dataset = UrbanSound8KDataset(audio_dir=audio_dir, file=train_metadata)
    val_dataset = UrbanSound8KDataset(audio_dir=audio_dir, file=val_metadata)
    test_dataset = UrbanSound8KDataset(audio_dir=audio_dir, file=test_metadata)
    
    return train_dataset, val_dataset, test_dataset


# Create DataLoaders for train, validation, and test datasets
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train(model, train_loader, criterion, optimizer, device):
    print("Starting training for the epoch...")
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Initialize tqdm for progress bar
    with tqdm(train_loader, desc="Training", unit="batch", ncols=100) as pbar:
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            # Flatten features for input to the ANN (if necessary)
            features = features.view(features.size(0), -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            # Update progress bar with current loss and accuracy
            avg_loss = running_loss / (pbar.n + 1)  # Average loss till now
            accuracy = (correct_preds / total_preds) * 100
            pbar.set_postfix(loss=avg_loss, accuracy=accuracy)

    # Return epoch's average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds * 100
    return avg_loss, accuracy




# Define testing/validation function
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = torch.tensor(batch['label']).to(device)

            # Flatten features for input to the ANN (if necessary)
            features = features.view(features.size(0), -1)

            # Forward pass
            outputs = model(features)

            # Calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    # Print statistics
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_preds / total_preds * 100
    return avg_loss, accuracy

def load_model(model_path, input_size, num_classes, device):
    """Load the trained model from the saved checkpoint."""
    model = AudioClassifierANN(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def make_inference(model, audio_file, sample_rate=22050, device='cpu'):
    
    waveform, sr = librosa.load(audio_file, sr=sample_rate)

    features = UrbanSound8KDataset(audio_dir='').extract_features(waveform)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    features_tensor = features_tensor.to(device)

    with torch.no_grad():
        outputs = model(features_tensor)

    _, predicted_class = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

    labels = ["air_conditioner", "car_horn", "children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]
    id = ([predicted_class.item()])[0]
    label_name = labels[int(id)]
    
    return label_name, confidence



if __name__ == '__main__':
    # Load datasets and DataLoaders
    # audio_dir = 'UrbanSound8K/audio'
    # file = 'UrbanSound8K/metadata/UrbanSound8K.csv'

    # # Split dataset into train, validation, and test sets
    # train_dataset, val_dataset, test_dataset = create_datasets(audio_dir, file)

    # # Create DataLoaders for each set
    # train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    # print(train_dataset[0]['features'].shape)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # # Initialize model, loss function, and optimizer
    # input_size = 40 + 32 + 128 + 1 + 1  # Features for MFCC, Chroma, Mel, ZCR, RMS
    # num_classes = len(train_dataset.label_encoder.classes_)  # Number of sound classes
    # model = AudioClassifierANN(input_size=input_size, num_classes=num_classes).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Initialize best validation accuracy and model checkpoint
    # best_val_acc = 0.0
    # best_model_path = 'best_model.pth'  # Path where the best model will be saved

    # # Training loop
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_acc = test(model, val_loader, criterion, device)
        
    #     print(f'Epoch {epoch+1}/{num_epochs}, '
    #           f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
    #           f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

    #     # Save the model if it has better validation accuracy
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f'Saved best model with Val Accuracy: {val_acc:.2f}%')

    # # Load the best model for final evaluation
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)

    # # Evaluate on the test set
    # test_loss, test_acc = test(model, test_loader, criterion, device)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    model = load_model("model_checkpoints/best_model_hamming.pth", input_size=202, num_classes=10, device='cpu')
    label_name, confidence = make_inference(model, audio_file = "UrbanSound8K/audio/fold3/6988-5-0-1.wav", sample_rate=22050, device='cpu')
    print(label_name, confidence)