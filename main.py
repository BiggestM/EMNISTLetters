import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_emnist_letters_csv(split='training'):
    """
    Load the EMNIST Letters dataset in CSV format.

    Parameters:
    - split: 'training' or 'testing' to specify the dataset split.

    Returns:
    - Tuple of NumPy arrays (images, labels).
    """
    if split == 'training':
        csv_file = 'Training/emnist-letters-train.csv'
    elif split == 'testing':
        csv_file = 'Testing/emnist-letters-test.csv'
    else:
        raise ValueError("Invalid split. Use 'training' or 'testing'.")

    # Load CSV data
    df = pd.read_csv(csv_file, header=None)

    # Extract labels and images
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values.reshape((len(df), 28, 28))

    return images, labels

# Example usage:
train_images, train_labels = load_emnist_letters_csv(split='training')
test_images, test_labels = load_emnist_letters_csv(split='testing')

# Split the training set into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)

# Preprocess the data
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Build and train the neural network model
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=.1)

# Flatten the images for MLP input
train_images_flat = train_images.reshape((train_images.shape[0], -1))
val_images_flat = val_images.reshape((val_images.shape[0], -1))
test_images_flat = test_images.reshape((test_images.shape[0], -1))

# Scale the data for better convergence
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images_flat)
val_images_scaled = scaler.transform(val_images_flat)
test_images_scaled = scaler.transform(test_images_flat)

# Train the model on the training set
model.fit(train_images_scaled, train_labels)

# Make predictions on the validation set
val_predictions = model.predict(val_images_scaled)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')

# Make predictions on the test set
test_predictions = model.predict(test_images_scaled)

# Evaluate the model on the test set
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
