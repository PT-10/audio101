import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the data
# Replace 'your_file.csv' with the actual file name
data = pd.read_csv('./data/features/combined_data.csv')

# Separate features and labels
X = data.iloc[:, 1:-1] 
y = data['source_file']

print("Test train split")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Fit transform complete")
svm_model = SVC(kernel='rbf', random_state=42)  
svm_model.fit(X_train, y_train)
print("Model fit complete")
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("/nClassification Report:/n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))