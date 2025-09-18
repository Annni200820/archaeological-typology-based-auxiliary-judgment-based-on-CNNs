import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd

# Import from utils
from utils.dataset import CustomDataset
from utils.preprocessing import transform, rev_label_map
from utils.model import SimpleCNN

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configuration
data_dir = '../data/raw/sample'
labels_file = '../data/raw/label.xlsx'
model_save_path = '../models/model.pth'
results_save_path = '../data/processed/test_results.xlsx'

# Create dataset
dataset = CustomDataset(directory=data_dir, labels_file=labels_file, transform=transform)

# Calculate sizes for each dataset split
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Split dataset using random_split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate model, define loss function and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model():
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Remove samples with label -1
            valid_mask = labels != -1
            if valid_mask.sum() == 0:
                continue  # Skip batch if all samples are removed
            images = images[valid_mask]
            labels = labels[valid_mask]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print('Training complete! Model saved to', model_save_path)


# Evaluation function
def evaluate_model():
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Prepare test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Make predictions
    predictions = []
    true_labels = []
    image_names = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            image_names.extend([os.path.basename(img_path).split('.')[0] for img_path in test_dataset.images])

    # Convert to readable labels
    predicted_labels = [rev_label_map[label] for label in predictions]

    # Save to Excel
    df = pd.DataFrame({
        'Image Name': image_names,
        'True Label': [rev_label_map[label] for label in true_labels],
        'Predicted Label': predicted_labels
    })
    df.to_excel(results_save_path, index=False)

    print('Prediction results have been saved to', results_save_path)


if __name__ == "__main__":
    # Uncomment to train the model
    # train_model()

    # Evaluate the model
    evaluate_model()