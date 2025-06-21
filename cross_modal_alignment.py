# %%
!mkdir -p /root/.kaggle
!mv /content/kaggle.json /root/.kaggle/kaggle.json

# %%
!mkdir -p /content/data/conformer

!kaggle kernels output tariq9mehmood9/eeg-classification-baseline -p /content/data/conformer
!kaggle datasets download tariq9mehmood9/eeg-visual-classification-new
!kaggle datasets download tariq9mehmood9/imagenet-40

!unzip -nq /content/eeg-visual-classification-new.zip -d /content/data
!unzip -nq /content/imagenet-40.zip -d /content/data/

# %%
!pip install git+https://github.com/openai/CLIP.git
!pip install -q braindecode

# %%
%pip install -q torchinfo
from torchinfo import summary

# %%
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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
from PIL import Image
import glob
from braindecode.models import EEGConformer
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed=45
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cpu_generator=torch.Generator(device='cpu').manual_seed(seed)

WORK_DIR = '/content/drive/MyDrive/thesis/thesis/s4_CA_GA_v2'
# WORK_DIR = '/content/drive/MyDrive/thesis/s4_CA_GA_v2'
# WORK_DIR = '/content/'
DATA_DIR = '/content/data'

# %%
data=torch.load(f"{DATA_DIR}/eeg_55_95_std.pth",weights_only=True)
splits=torch.load(f"{DATA_DIR}/block_splits_by_image_all.pth",weights_only=True)
captions_with_bbox_data=torch.load(f"{DATA_DIR}/captions_with_bbox_data.pth",weights_only=True)

captions_data={"captions": [], "bbox_labels": [], "images": []}
for i in range(len(captions_with_bbox_data['images'])):
    caption=captions_with_bbox_data['captions_with_bbox'][i]['<CAPTION>']
    bbox_label=captions_with_bbox_data['captions_with_bbox'][i]['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
    image=captions_with_bbox_data['images'][i]
    captions_data["captions"].append(caption)
    captions_data["bbox_labels"].append(bbox_label)
    captions_data["images"].append(image)

def generate_helper_mappings():
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

    label_to_class=dict()
    label_to_simple_class=dict()
    for idx,label in enumerate(labels):
        imagenet_class=label_to_imagenet_classes[label]
        label_to_class[idx]=imagenet_class
        # selected labels to simplified class names used in paper
        s=re.split(r"[ ,!;.-]+", imagenet_class)
        simplified_class = list(set([c for c in s if c in classes_in_paper]))
        if len(simplified_class) == 0:
            print("No possible class found for: ", imagenet_class)
            print(s)
        elif len(simplified_class) > 1:
            print("multiple possible classes for: ", imagenet_class)
        else:
            label_to_simple_class[idx]=simplified_class[0]
        # convert jack-o-lantern class name to pumpkin, which captures the semantic meaning and is still simplified. (the paper uses "jack")
        label_to_simple_class[labels.index('n03590841')] = "pumpkin"
        # TODO: update imagenet label for "iron" and "sorrel" to perhaps "smoothing iron" and "sorrel horse" respectively

    image_to_class={i:label_to_class[labels.index(image_name.split('_')[0])] for i,image_name in enumerate(data['images'])}
    image_to_simple_class={i:label_to_simple_class[labels.index(image_name.split('_')[0])] for i,image_name in enumerate(data['images'])}
    image_to_name={i:image_name for i,image_name in enumerate(data['images'])}
    image_to_path={i: image.split('_')[0] + "/" + image + ".JPEG" for i,image in enumerate(data['images'])}

    assert label_to_imagenet_classes[labels[10]] == "canoe"
    assert label_to_class[10] == "canoe"
    assert label_to_simple_class[10] == "canoe"
    assert image_to_class[618] == "canoe"
    assert image_to_simple_class[618] == "canoe"
    assert all(os.path.exists(f"{DATA_DIR}/imageNet_images/{image}") for image in image_to_path.values()), \
            f"Some paths don't exist: {[image for image in image_to_path.values() if not os.path.exists(f'{DATA_DIR}/imageNet_images/{image}')]}"

    return label_to_class, label_to_simple_class, image_to_class, image_to_simple_class, image_to_name, image_to_path

label_to_class, label_to_simple_class, image_to_class, image_to_simple_class, image_to_name, image_to_path = generate_helper_mappings()

captions_df = pd.DataFrame(captions_data)
dataset_df=pd.DataFrame(data["dataset"])
dataset_df["class"] = dataset_df["label"].map(label_to_simple_class)

df=dataset_df.join(captions_df[["captions", "bbox_labels"]], on="image")[["subject", "eeg", "captions", "bbox_labels", "class", "label", "image"]]

# %%
df

# %%
data = pd.read_csv("/content/captions_info.csv")
print(data.head())

# %%
df_merged = df.merge(
    data[['image_idx', 'caption']],  # only need image_idx and new caption
    how='left',
    left_on='image',                 # column in df
    right_on='image_idx'             # column in data
)

# %%
df_merged['captions'] = df_merged['caption']
df_merged.drop(['image_idx', 'caption'], axis=1, inplace=True)

# %%
df_merged.head()

# %%
df = df_merged

# %%
# Handle duplicates
df["bbox_labels"] = df["bbox_labels"].apply(
    lambda labels: list(dict.fromkeys(labels))
)

# %%
df

# %%
train_split=splits["splits"][0]["train"]
val_split=splits["splits"][0]["val"]
test_split=splits["splits"][0]["test"]

train_df=df.iloc[train_split]
val_df=df.iloc[val_split]
test_df=df.iloc[test_split]

# %% [markdown]
# # **Dataloader**

# %%
class EEGTextDataset(Dataset):
    def __init__(self, df, tl=20, th=460):
        """
        df: exploded dataframe, each row has exactly 1 bounding-box label.
        """
        self.df = df.reset_index(drop=True)
        self.tl = tl
        self.th = th

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_tensor = row["eeg"]  # shape [n_chans, n_times]
        eeg_tensor = eeg_tensor[:, self.tl:self.th] # shape: [128, 440]

        # bbox_label = row["bbox_labels"]
        # concept = row["class"]
        caption = row["captions"]

        return eeg_tensor, caption

# %%
train_dataset = EEGTextDataset(train_df)
val_dataset   = EEGTextDataset(val_df)
test_dataset  = EEGTextDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
for eeg_batch, text_batch in test_loader:
    print("Example text_batch:", text_batch[:15])
    break

# %% [markdown]
# # **Precompute Unique Caption Embeddings**

# %%
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# Collect all unique captions
all_captions = pd.concat([train_df["captions"], val_df["captions"], test_df["captions"]])
unique_captions = all_captions.unique()

caption2embed = {}
with torch.no_grad():
    for cap in tqdm(unique_captions, desc="Precomputing text embeddings"):
        # Tokenize and encode
        tokens = clip.tokenize(cap, truncate=True).to(device)
        text_emb = clip_model.encode_text(tokens).float()
        text_emb = F.normalize(text_emb, dim=-1)  # shape [1, embed_dim]

        caption2embed[cap] = text_emb.squeeze(0).cpu()  # store on CPU

# %%
caption2embed['Orange jack-o’-lantern with a carved face.'].shape

# %% [markdown]
# # **Load Encoders**

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# %%
model = EEGConformer(
    n_outputs=40,
    n_chans=128,
    n_times=440,
    n_filters_time=40,
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    final_fc_length=920,
    add_log_softmax=False,
    return_features=True
)

# Modify final layers to produce 768 features
model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=40, bias=True)

# Remove final classification layer, leaving us with 768-d embeddings
model.final_layer = nn.Identity()

model = model.to(device)

# %%
summary(model, input_size=(1, 128, 440))

# %%
def clip_style_contrastive_loss(eeg_embeds, text_embeds, temperature=0.07):
    """
    eeg_embeds: [batch_size, embed_dim]
    text_embeds: [batch_size, embed_dim]
    """
    batch_size = eeg_embeds.size(0)

    # Similarity matrices
    logits_eeg = eeg_embeds @ text_embeds.t() / temperature     # [B, B]
    logits_text = text_embeds @ eeg_embeds.t() / temperature    # [B, B]

    labels = torch.arange(batch_size, device=eeg_embeds.device) # [0..B-1]

    # InfoNCE loss (two-way)
    loss_eeg = F.cross_entropy(logits_eeg, labels)
    loss_text = F.cross_entropy(logits_text, labels)
    loss = (loss_eeg + loss_text) / 2.0

    # Accuracy metrics (in-batch retrieval)
    acc_eeg = (logits_eeg.argmax(dim=-1) == labels).float().mean()
    acc_text = (logits_text.argmax(dim=-1) == labels).float().mean()

    metrics = {
        "loss": loss.item(),
        "acc_eeg": acc_eeg.item(),
        "acc_text": acc_text.item(),
    }
    return loss, metrics

# %%
# Freeze CLIP
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
num_epochs = 100
temperature = 0.07

# Early-stopping config
patience = 15
best_val_loss = float('inf')
early_stop_counter = 0
checkpoint_path = "best_model.pth"

for epoch in range(num_epochs):
    ###### TRAINING PHASE ######
    model.train()
    train_losses = []
    train_accs_eeg = []
    train_accs_text = []

    # tqdm progress bar for training
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for eeg_batch, text_batch in train_pbar:
        eeg_batch = eeg_batch.to(device)

        # Encode text (frozen CLIP)
        with torch.no_grad():
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            text_embeds = clip_model.encode_text(text_tokens).float()
            text_embeds = F.normalize(text_embeds, dim=-1)

        # Get text embeddings from dictionary
        # text_embeds = []
        # for cap in text_batch:
        #     text_embeds.append(caption2embed[cap])  # [768] on CPU
        # text_embeds = torch.stack(text_embeds, dim=0).to(device)  # [B, 768]

        # Encode EEG (trainable)
        eeg_embeds = model(eeg_batch)
        eeg_embeds = F.normalize(eeg_embeds, dim=-1)

        # Contrastive loss
        loss, metrics = clip_style_contrastive_loss(eeg_embeds, text_embeds, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_losses.append(loss.item())
        train_accs_eeg.append(metrics["acc_eeg"])
        train_accs_text.append(metrics["acc_text"])

        # Show some info in progress bar
        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc_eeg": f"{metrics['acc_eeg']:.4f}",
            "acc_text": f"{metrics['acc_text']:.4f}"
        })

    mean_train_loss = np.mean(train_losses)
    mean_train_acc_eeg = np.mean(train_accs_eeg)
    mean_train_acc_text = np.mean(train_accs_text)

    ###### VALIDATION PHASE ######
    model.eval()
    val_losses = []
    val_accs_eeg = []
    val_accs_text = []

    # tqdm progress bar for validation
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for eeg_batch, text_batch in val_pbar:
            eeg_batch = eeg_batch.to(device)

            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            text_embeds = clip_model.encode_text(text_tokens).float()
            text_embeds = F.normalize(text_embeds, dim=-1)

            eeg_embeds = model(eeg_batch)
            eeg_embeds = F.normalize(eeg_embeds, dim=-1)

            val_loss, val_metrics = clip_style_contrastive_loss(eeg_embeds, text_embeds, temperature)
            val_losses.append(val_loss.item())
            val_accs_eeg.append(val_metrics["acc_eeg"])
            val_accs_text.append(val_metrics["acc_text"])

    mean_val_loss = np.mean(val_losses)
    mean_val_acc_eeg = np.mean(val_accs_eeg)
    mean_val_acc_text = np.mean(val_accs_text)

    # Print epoch summary
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"TrainLoss: {mean_train_loss:.4f}, ValLoss: {mean_val_loss:.4f} | "
          f"TrainAcc_EEG: {mean_train_acc_eeg:.4f}, ValAcc_EEG: {mean_val_acc_eeg:.4f} | "
          f"TrainAcc_Text: {mean_train_acc_text:.4f}, ValAcc_Text: {mean_val_acc_text:.4f}")

    ###### EARLY STOPPING & CHECKPOINTING ######
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  --> Saved new best model at epoch {epoch+1} (val_loss={best_val_loss:.4f})")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

print(f"Training complete. Best val_loss={best_val_loss:.4f}")

# model.load_state_dict(torch.load(checkpoint_path))
