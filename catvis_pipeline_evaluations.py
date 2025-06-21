# %%
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import re
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import hashlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed=45
np.random.seed(seed)
torch.manual_seed(seed)
cpu_generator=torch.Generator(device='cpu').manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    cuda_generator=torch.Generator(device=device).manual_seed(seed)

WORK_DIR = '/workspace/work'
DATA_DIR = '/workspace/data'

# %% [markdown]
# # Evaluation Metrics

# %% [markdown]
# ## Classification Metrics (top-k, F1-score)

# %%
with open(f"{WORK_DIR}/all_labels.pkl", "rb") as f:
    all_labels=pickle.load(f)

with open(f"{WORK_DIR}/all_predictions.pkl", "rb") as f:
    all_predictions=pickle.load(f)

# %%
from sklearn.metrics import f1_score

def compute_metrics(all_labels, all_predictions, top_k=5):
    """
    Compute top-1 to top-k accuracy and F1 score.

    Parameters:
        all_labels (list): Ground truth labels.
        all_predictions (list of lists): Each sublist contains predicted probabilities for each class.
        top_k (int): Maximum value of k for top-k accuracy.

    Returns:
        dict: Top-k accuracy for k=1 to top_k and F1 score.
    """
    # Convert to tensors
    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)  # Shape: (N, num_classes)

    # Compute top-k predictions
    top_k_preds = all_predictions.topk(k=top_k, dim=1).indices  # Shape: (N, top_k)

    # Compute top-k accuracy
    top_k_accuracies = {}
    for k in range(1, top_k + 1):
        correct = (top_k_preds[:, :k] == all_labels.unsqueeze(1)).any(dim=1)  # Correct predictions in top-k
        top_k_accuracies[f"top-{k} accuracy"] = correct.float().mean().item() * 100

    # Compute F1 score (use top-1 predictions for F1)
    top_1_preds = top_k_preds[:, 0]  # Shape: (N,)
    f1 = f1_score(all_labels.cpu(), top_1_preds.cpu(), average="weighted")  # Weighted F1-score

    # Add F1 score to metrics
    top_k_accuracies["F1 score"] = f1 * 100

    return top_k_accuracies


metrics = compute_metrics(all_labels, all_predictions, top_k=5)
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}%")

# %%
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

true_labels = np.array(all_labels)
predicted_labels = np.argmax(all_predictions, axis=1)
classes = [label_to_simple_class[label] for label in range(40)]
assert classes[10] == 'canoe'

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Normalize the confusion matrix for better readability
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_normalized,
    annot=False,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    xticklabels=classes,
    yticklabels=classes,
    square=True,
    linewidths=0.5
)

# Add axis labels and title
ax.set_xlabel("Predicted Labels", fontsize=14)
ax.set_ylabel("True Labels", fontsize=14)
ax.set_title("Confusion Matrix", fontsize=16, pad=20)

# Rotate axis tick labels for better readability
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Adjust layout for saving
plt.tight_layout()

plt.savefig(f"{WORK_DIR}/confusion_matrix.png", dpi=300)
plt.show()

# %%
# # Basic data analysis
# accuracy = accuracy_score(true_labels, predicted_labels)
# print(f"Accuracy: {accuracy:.4f}\n")

# print("Classification Report:")
# print(classification_report(true_labels, predicted_labels))

# %% [markdown]
# ## *N*-way Top-*K* Classification Accuracy of Generation (GA)

# %%
!pip install -q torchmetrics

# %%
from torchvision.models import ViT_H_14_Weights, vit_h_14
from torchmetrics.functional import accuracy

def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        # The correct class (class_id) is placed at index 0 in pred_picked
        # Therefore, the ground truth for this reduced setup is always 0
        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device),task="multiclass",num_classes=n_way,
                    top_k=top_k)
        acc_list.append(acc.item())

    # print(np.mean(acc_list))
    return np.mean(acc_list), np.std(acc_list)

weights=ViT_H_14_Weights.DEFAULT
vit_clf_model = vit_h_14(weights=weights)
preprocess = weights.transforms()
vit_clf_model = vit_clf_model.to(device)
vit_clf_model = vit_clf_model.eval()
n_way = 50
num_trials = 50
top_k = 1

acc_list = []
gt_folder = f"{WORK_DIR}/picture-gene-onlygt"
gene_folder= f"{WORK_DIR}/picture-gene"
gt_images_name = os.listdir(gt_folder)
gt_images_name.sort()

# 1995_shoe.jpg
# 1995_shoe_s4_cat_0.png
for subject in test_df['subject'].unique():
    # print(f"Subject: {subject}")
    sub_acc_list = []
    for gt_name in gt_images_name:
        # print(gt_name)

        # Load GT image and the path of genetrated images
        real_image = Image.open(gt_folder + "/" + gt_name).convert('RGB')
        generated_images = glob.glob(f"{gene_folder}/{gt_name.split('.')[0]}_s{subject}_*.png")

        gt = preprocess(real_image).unsqueeze(0).to(device)
        gt_class_id = vit_clf_model(gt).squeeze(0).softmax(0).argmax().item()

        # Evaluate
        for generated_image_path in generated_images:
            generated_image = Image.open(generated_image_path).convert('RGB')
            pred = preprocess(generated_image).unsqueeze(0).to(device)
            pred_out = vit_clf_model(pred).squeeze(0).softmax(0).detach()
            acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
            acc_list.append(acc)
            sub_acc_list.append(acc)

        # print(f"Running Accuracy: {np.mean(acc_list)*100:.2f}%")
    print(f"Subject {subject} Accuracy: {np.mean(sub_acc_list)*100:.2f}%")
print(f"Average Accuracy: {np.mean(acc_list)*100:.2f}%")

# %% [markdown]
# ## Inception Score (IS)

# %%
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

gen_img_dir=f"{WORK_DIR}/picture-gene"
batch_size=32

# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]

def readDir(dirPath=gen_img_dir):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            f = os.path.join(dirPath, f)
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles.extend(subFiles)
            else:
                if "_gt" not in f:
                    allFiles.append(f)
        return allFiles
    else:
        return 'Error, not a dir'

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

def inception_score(batch_size=batch_size, resize=True):
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')

    files = readDir()
    N = len(files)
    preds = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))

    for i in tqdm(range(0, N, batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        y = get_pred(batch)
        preds[i:i + batch_size] = y

    assert batch_size > 0
    assert N > batch_size

    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    py = np.mean(preds, axis=0)  # marginal probability
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence

    mean_kl = np.mean(scores)
    inception_score = np.exp(mean_kl)

    return inception_score

IS = inception_score()
print('The Inception Score is %.4f' % IS)

# %% [markdown]
# ## Fréchet Inception Distance (FID)

# %%
!pip install -q pytorch-fid

# %%
import torchvision.transforms as transforms
from pytorch_fid import fid_score

gen_img_dir=f"{WORK_DIR}/picture-gene"
gt_img_dir =f"{WORK_DIR}/picture-gene-onlygt"
temp_path  =f"{WORK_DIR}/temppath"

# Define the transform to resize the image to 512x512
transform = transforms.Compose([
    transforms.Resize((512, 512))
])

# Save the transformed gt_images to temppath
os.makedirs(temp_path, exist_ok=True)
for filename in os.listdir(gt_img_dir):
    src_path = os.path.join(gt_img_dir, filename)
    dest_path= os.path.join(temp_path, filename)
    with Image.open(src_path) as img:
        transformed_img = transform(img)
        transformed_img.save(dest_path)

fid_value = fid_score.calculate_fid_given_paths([gen_img_dir, temp_path], batch_size=50, device=device, dims=2048)

print('FID:', fid_value)

shutil.rmtree(temp_path)

# %% [markdown]
# # Visual

# %%
from PIL import Image, ImageDraw, ImageFont
from IPython import display
import os, re, glob

def create_image_grid(selected_images, subject=4, grid_cols=5, image_size=(128, 128), spacing=5, group_spacing=20):
    """
    Create a compact grid of images using PIL for bird's eye view.

    Parameters:
        selected_images (list of str): List of selected image names (e.g., "n02106662_6706.jpeg").
        grid_cols (int): Number of columns in the grid (group of GT and generated samples).
        image_size (tuple): Size (width, height) to resize images for the grid.
        spacing (int): Space between images within a group.
        group_spacing (int): Space between groups of images (across different groups).

    Returns:
        PIL Image: The resulting grid image.
    """
    # Total images per group (GT + 2 samples)
    images_per_group = 3
    num_groups = len(selected_images)

    # Calculate grid size
    grid_rows = (num_groups + grid_cols - 1) // grid_cols
    grid_width = grid_cols * images_per_group * image_size[0] + (grid_cols - 1) * spacing + (grid_cols - 1) * group_spacing
    grid_height = grid_rows * (image_size[1] + spacing) + (grid_rows - 1) * group_spacing + 50  # Extra space for labels

    # Create a blank canvas
    grid_image = Image.new("RGB", (grid_width, grid_height), "white")

    # Create a drawing object
    draw = ImageDraw.Draw(grid_image)

    # Font for text
    font = ImageFont.load_default(size=40)

    # Draw dashed vertical lines after every column
    line_color = (64, 64, 64)  # Black color for the lines
    line_width = 5  # Line width
    dash_length = 20  # Length of each dash

    for col in range(1, grid_cols):
        x_pos = col * images_per_group * image_size[0] + col * spacing + col * group_spacing - group_spacing // 2 + 1
        for y in range(0, grid_height, dash_length * 2):  # Drawing the dashes
            draw.line((x_pos, y-70, x_pos, min(y-70 + dash_length, grid_height)), fill=line_color, width=line_width)

    # Load and paste images into the grid
    for i, image_name in enumerate(selected_images):
        group_col = i % grid_cols  # Column index for this group
        group_row = i // grid_cols  # Row index for this group

        # X and Y offsets for images
        x_offset = group_col * images_per_group * image_size[0] + group_col * spacing + group_col * group_spacing
        y_offset = group_row * (image_size[1] + spacing) + group_row * group_spacing

        # Paths for original and generated images
        gt_path = os.path.join(WORK_DIR, "picture-gene-onlygt", image_name)
        generated_path_1 = glob.glob(f"{WORK_DIR}/picture-gene/{image_name.split('.')[0]}_s{subject}_*_0.png")[0]
        generated_path_2 = glob.glob(f"{WORK_DIR}/picture-gene/{image_name.split('.')[0]}_s{subject}_*_1.png")[0]

        try:
            # Load and resize images
            gt_image = Image.open(gt_path).resize(image_size).convert("RGB")
            gen_image_1 = Image.open(generated_path_1).resize(image_size).convert("RGB")
            gen_image_2 = Image.open(generated_path_2).resize(image_size).convert("RGB")

            # Paste images into grid
            grid_image.paste(gt_image, (x_offset, y_offset))
            grid_image.paste(gen_image_1, (x_offset + image_size[0] + spacing, y_offset))
            grid_image.paste(gen_image_2, (x_offset + 2 * (image_size[0] + spacing), y_offset))

            # Draw a red bounding box around the first image in the group (ground truth)
            draw.rectangle(
                [x_offset, y_offset, x_offset + image_size[0], y_offset + image_size[1]],
                outline="red",
                width=5
            )
        except FileNotFoundError as e:
            print(f"Error: {e} for {image_name}")
            continue

    # Add labels only at the bottom row of the canvas
    text_y_offset = grid_height - 30  # Position for text at the very bottom
    for col in range(grid_cols):
        # X-offset for each group
        x_offset = col * images_per_group * image_size[0] + col * spacing + col * group_spacing

        draw.text((x_offset + image_size[0] // 2, text_y_offset), "GT", fill="black", font=font, anchor="mm")
        draw.text((x_offset + image_size[0] + spacing + image_size[0] // 2, text_y_offset), "Sample 1", fill="black", font=font, anchor="mm")
        draw.text((x_offset + 2 * (image_size[0] + spacing) + image_size[0] // 2, text_y_offset), "Sample 2", fill="black", font=font, anchor="mm")

    return grid_image

# %%
selected_images = [
    '0901_cat.jpg',
    '1336_canoe.jpg',
    '0838_computer.jpg',

    '0604_parachute.jpg',
    '0193_pool.jpg',
    '0040_piano.jpg',

    '0102_parachute.jpg',
    '0749_guitar.jpg',
    '0964_bike.jpg',

    '0467_broom.jpg',
    '0435_golf.jpg',
    '0032_locomotive.jpg',

    '0801_mug.jpg',
    '0858_dog.jpg',
    '0053_chair.jpg',

    '0745_convertible.jpg',
    '0219_chair.jpg',
    '0981_cat.jpg'
 ]

create_image_grid(selected_images, grid_cols=3, image_size=(300, 300), spacing=5, group_spacing=30)


