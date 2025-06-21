# %%
%pip install -q braindecode

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed=45
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch_generator = torch.Generator()
torch_generator.manual_seed(seed)

WORK_DIR = '/kaggle/working'
DATA_DIR = '/kaggle/input/eeg-visual-classification-new'
# DATA_DIR = WORK_DIR

from braindecode.models import EEGConformer

# %%
data=torch.load(f"{DATA_DIR}/eeg_55_95_std.pth",weights_only=True)
splits=torch.load(f"{DATA_DIR}/block_splits_by_image_all.pth",weights_only=True)
captions_data=torch.load(f"{DATA_DIR}/captions_data.pth",weights_only=True)

labels = data["labels"] # list of 40 sysnet labels
classes_in_paper=["dog", "cat", "butterfly", "sorrel", "capuchin", "elephant", "panda", "fish", "airliner", "broom", "canoe", "phone", "mug", "convertible", "computer", "watch", "guitar", "locomotive", "espresso", "chair", "golf", "piano", "iron", "jack", "mailbag", "missile", "mitten", "bike", "tent", "pajama", "parachute", "pool", "radio", "camera", "gun", "shoe", "banana", "pizza", "daisy", "bolete"]

with open(f"{DATA_DIR}/imagenet_class_labels.txt", "r") as f:
    lines=f.read().split("\n")

# 1K labels to all possible class values
label_to_imagenet_classes = dict()
for line in lines:
    s=line.split(': ')
    try:
        label_to_imagenet_classes[s[0]] = s[1]
    except:
        label_to_imagenet_classes[s[0]] = None

# selected labels to simplified class names used in paper
label_to_class=dict()
for idx,label in enumerate(labels):
    imagenet_class=label_to_imagenet_classes[label]
    s=re.split(r"[ ,!;.-]+", imagenet_class)
    simplified_class = list(set([c for c in s if c in classes_in_paper]))
    if len(simplified_class) == 0:
        print("No possible class found for: ", imagenet_class)
        print(s)
    elif len(simplified_class) > 1:
        print("multiple possible classes for: ", imagenet_class)
    else:
        label_to_class[idx]=simplified_class[0]

# convert jack-o-lantern class name to pumpkin, which captures the semantic meaning and is still simplified. (the paper uses "jack")
label_to_class[labels.index('n03590841')] = "pumpkin"

# image "n03452741_17620.JPEG" contains invalid image and consequently null captions
captions_data["captions"][180] = "A grand piano."
captions_data["detailed_captions"][180] = "The image shows a grand piano."

captions_df = pd.DataFrame(captions_data)
captions_df

dataset_df=pd.DataFrame(data["dataset"])
dataset_df["class"] = dataset_df["label"].map(label_to_class)
assert label_to_imagenet_classes[labels[10]] == "canoe"
dataset_df

df=dataset_df.join(captions_df[["captions"]], on="image")[["eeg", "captions", "class", "label"]]

# %%
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

train_split=splits["splits"][0]["train"]
val_split=splits["splits"][0]["val"]
test_split=splits["splits"][0]["test"]

train_df=df.iloc[train_split]
val_df=df.iloc[val_split]
test_df=df.iloc[test_split]

train_df

# %%
# 20ms - 460ms as per the paper
tl=20
th=460
eeg_train = np.array(train_df['eeg'].apply(lambda x: x[:, tl:th]).tolist())
train_labels = train_df['label'].tolist()
print("eeg_train.shape: ", eeg_train.shape)

eeg_val = np.array(val_df['eeg'].apply(lambda x: x[:, tl:th]).tolist())
val_labels = val_df['label'].tolist()
print("eeg_val.shape: ", eeg_val.shape)

eeg_test = np.array(test_df['eeg'].apply(lambda x: x[:, tl:th]).tolist())
test_labels = test_df['label'].tolist()
print("eeg_test.shape: ", eeg_test.shape)

# %%
batch_size = 128
train_dataset = TensorDataset(torch.tensor(eeg_train, dtype=torch.float32),
                              torch.tensor(train_labels, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_generator)

val_dataset = TensorDataset(torch.tensor(eeg_val, dtype=torch.float32),
                            torch.tensor(val_labels, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(torch.tensor(eeg_test, dtype=torch.float32),
                            torch.tensor(test_labels, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
model = EEGConformer(n_outputs=40, n_chans=128, n_times=440, n_filters_time=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15, final_fc_length=920, add_log_softmax=False, return_features=True)
# set the feature dimension to match clip embeddings dimension
model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=40, bias=True)
model.to(device)
input_shape = (128, 440)
# summary(model, input_shape, batch_dim=0)
# print(model)

# %%
from tqdm.notebook import tqdm
import torch

NUM_EPOCHS = 120
MODEL_SAVE_PATH = 'best_model.pth'

loss_list = []
accuracy_list = []

val_loss_list = []
val_accuracy_list = []

best_val_acc = 0.0

classification_loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()
for epoch in tqdm(range(NUM_EPOCHS)):
    running_loss = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader):
        eeg, label = data
        eeg, label = eeg.to(device), label.to(device)

        optimizer.zero_grad()
        outputs, eeg_embeddings = model(eeg)
        loss = classification_loss(outputs, label)
        loss.backward()

        optimizer.step()

        correct += (outputs.argmax(dim=-1) == label).float().sum().item()
        
        running_loss += loss.item()
        
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    loss_list.append(avg_loss)
    accuracy = 100 * correct / len(train_dataset)
    accuracy_list.append(accuracy)

    # Validation phase
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    val_correct = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for val_data in val_loader:
            eeg_val, label_val = val_data
            eeg_val, label_val = eeg_val.to(device), label_val.to(device)

            outputs_val, eeg_embeddings_val = model(eeg_val)
            batch_val_loss = classification_loss(outputs_val, label_val)

            val_loss += batch_val_loss.item()
            val_correct += (outputs_val.argmax(dim=-1) == label_val).float().sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)
    
    val_accuracy = 100 * val_correct / len(val_dataset)
    val_accuracy_list.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
          f'TrL: {avg_loss:.4f}, VaL: {avg_val_loss:.4f}, '
          f'TrA: {accuracy:.2f}%, VaA: {val_accuracy:.2f}%')

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save the best model
        print(f'Best model saved at epoch {epoch + 1}')

# load the best model for further evaluation or inference
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(loss_list, label="Train Loss", color="blue", linewidth=2)
axes[0].plot(val_loss_list, label="Validation Loss", color="red", linewidth=2)

axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(accuracy_list, label="Train Accuracy", color="blue", linewidth=2)
axes[1].plot(val_accuracy_list, label="Validation Accuracy", color="red", linewidth=2)

axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# Test phase
model.eval()  # Switch to evaluation mode
test_loss = 0.0
test_correct = 0.0

# Lists to store ground truth and predictions
all_labels = []
all_predictions = []

with torch.no_grad():  # Disable gradient calculation
    for test_data in test_loader:
        eeg_test, label_test = test_data
        eeg_test, label_test = eeg_test.to(device), label_test.to(device)

        outputs_test, eeg_embeddings_test = model(eeg_test)
        batch_test_loss = classification_loss(outputs_test, label_test)

        test_loss += batch_test_loss.item()
        test_correct += (outputs_test.argmax(dim=-1) == label_test).float().sum().item()

        # Store predictions and labels
        all_labels.extend(label_test.cpu().numpy())
        all_predictions.extend(outputs_test.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)

test_accuracy = 100 * test_correct / len(test_dataset)

# Print test results
print(f'Test Results: '
      f'Test Loss: {avg_test_loss:.4f}, '
      f'Test Accuracy: {test_accuracy:.2f}%')

# %%
with open(f"{WORK_DIR}/all_labels.pkl", "wb") as f:
    pickle.dump(all_labels, f)

with open(f"{WORK_DIR}/all_predictions.pkl", "wb") as f:
    pickle.dump(all_predictions, f)


