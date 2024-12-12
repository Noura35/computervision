import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset from pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Convert data and labels to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check data consistency: Ensure all feature vectors have the same length
# Get the shape of the data (number of samples, number of features)
print(f'Number of samples: {len(data)}')
print(f'Length of the first sample: {len(data[0])}')
print(f'Length of the last sample: {len(data[-1])}')

# Ensure the labels are consistent
unique_labels = np.unique(labels)
print(f'Unique labels: {unique_labels}')

# If data is ragged (variable-length), handle it by padding/trimming
# Check if there's inconsistency in the length of samples
max_len = max(len(sample) for sample in data)
data = np.array([np.pad(sample, (0, max_len - len(sample)), mode='constant') for sample in data])

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier with some parameters
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)

# Print the classification accuracy
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully to 'model.p'.")
