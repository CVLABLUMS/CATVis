# %%
!mkdir -p /root/.kaggle
!mv /workspace/kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

# %%
!pip install -q kaggle pandas matplotlib torchinfo braindecode
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q diffusers transformers huggingface_hub scipy ftfy accelerate

# %%
!mkdir -p /workspace/data/conformer
!kaggle kernels output tariq9mehmood9/eeg-classification-baseline -p /workspace/data/conformer

!kaggle datasets download tariq9mehmood9/eeg-visual-classification-new
!kaggle datasets download tariq9mehmood9/imagenet-40

!unzip -nq /workspace/eeg-visual-classification-new.zip -d /workspace/data
!unzip -nq /workspace/imagenet-40.zip -d /workspace/data/

# %% [markdown]
# # Dataset

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

# ref: https://github.com/RomGai/BrainVis/blob/4b0b20d704f512cd4331f6f6e9ac2d613749c4a2/cascade_diffusion.py#L17
prompt_dict = {
    'n02106662': 'german shepherd dog',
    'n02124075': 'Egyptian cat', #added Egyptian
    'n02281787': 'lycaenid butterfly',
    'n02389026': 'sorrel horse',
    'n02492035': 'white-faced capuchin (Cebus)', #removed Cebus capucinus
    'n02504458': 'African elephant',
    'n02510455': 'panda',
    'n02607072': 'anemone fish',
    'n02690373': 'airliner',
    'n02906734': 'broom',
    'n02951358': 'canoe or kayak',
    'n02992529': 'cellular telephone',
    'n03063599': 'coffee mug',
    'n03100240': 'old convertible',
    'n03180011': 'desktop computer',
    'n03197337': 'digital watch',
    'n03272010': 'electric guitar',
    'n03272562': 'electric locomotive',
    'n03297495': 'espresso maker',
    'n03376595': 'folding chair',
    'n03445777': 'golf ball',
    'n03452741': 'grand piano',
    'n03584829': 'smoothing iron',
    'n03590841': 'Orange jack-o’-lantern',
    'n03709823': 'mailbag',
    'n03773504': 'missile',
    'n03775071': 'mitten or glove',
    'n03792782': 'mountain bike', #removed all-terrain bike
    'n03792972': 'mountain tent',
    'n03877472': 'pajama',
    'n03888257': 'parachute',
    'n03982430': 'pool table or billiard table or snooker table', #added ors
    'n04044716': 'radio telescope',
    'n04069434': 'reflex camera',
    'n04086273': 'revolver, six-shooter',
    'n04120489': 'running shoe',
    'n07753592': 'banana',
    'n07873807': 'pizza',
    'n11939491': 'daisy',
    'n13054560': 'bolete mushroom' #added mushroom
}

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
        # label_to_class[idx]=imagenet_class
        label_to_class[idx]=prompt_dict[label]
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

    image_to_class={i:label_to_class[labels.index(image_name.split('_')[0])] for i,image_name in enumerate(data['images'])}
    image_to_simple_class={i:label_to_simple_class[labels.index(image_name.split('_')[0])] for i,image_name in enumerate(data['images'])}
    image_to_name={i:image_name for i,image_name in enumerate(data['images'])}
    image_to_path={i: image.split('_')[0] + "/" + image + ".JPEG" for i,image in enumerate(data['images'])}

    assert label_to_imagenet_classes[labels[10]] == "canoe"
    assert label_to_class[10] == "canoe" or "canoe or kayak"
    assert label_to_simple_class[10] == "canoe"
    assert image_to_class[618] == "canoe" or "canoe or kayak"
    assert image_to_simple_class[618] == "canoe"
    assert all(os.path.exists(f"{DATA_DIR}/imageNet_images/{image}") for image in image_to_path.values()), \
            f"Some paths don't exist: {[image for image in image_to_path.values() if not os.path.exists(f'{DATA_DIR}/imageNet_images/{image}')]}"

    return label_to_class, label_to_simple_class, image_to_class, image_to_simple_class, image_to_name, image_to_path

label_to_class, label_to_simple_class, image_to_class, image_to_simple_class, image_to_name, image_to_path = generate_helper_mappings()

captions_df = pd.DataFrame(captions_data)
dataset_df=pd.DataFrame(data["dataset"])
dataset_df["class"] = dataset_df["label"].map(label_to_class)

df=dataset_df.join(captions_df[["captions", "bbox_labels"]], on="image")[["subject", "eeg", "captions", "bbox_labels", "class", "label", "image"]]

# %%
# train_split=splits["splits"][0]["train"]
# val_split=splits["splits"][0]["val"]
test_split=splits["splits"][0]["test"]

# train_df=df.iloc[train_split]
# val_df=df.iloc[val_split]
test_df=df.iloc[test_split]

# %% [markdown]
# # Conformer-Based EEG Classifier

# %%
from braindecode.models import EEGConformer

model = EEGConformer(n_outputs=40, n_chans=128, n_times=440, n_filters_time=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15, final_fc_length=920, add_log_softmax=False, return_features=True)
# set the feature dimension to match clip embeddings dimension
model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=40, bias=True)
model.to(device)

classification_loss = torch.nn.CrossEntropyLoss()

MODEL_SAVE_PATH=f"{DATA_DIR}/conformer/best_model.pth"
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# %% [markdown]
# # CLIP Aligned EEG Encoder for Retrieval

# %%
import clip

clip_model, preprocess = clip.load("ViT-L/14", device=device)

cnt_model = EEGConformer(
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
cnt_model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
cnt_model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=40, bias=True)

# Remove final classification layer, leaving us with 768-d embeddings
cnt_model.final_layer = nn.Identity()

cnt_model = cnt_model.to(device)

cnt_model.load_state_dict(torch.load(f"{DATA_DIR}/ckpt_cnt_model.pth"))

# %% [markdown]
# # Retrieval Pre-requisites

# %%
from torch.utils.data import Dataset, DataLoader

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

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels, images, subjects, captions):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.images = torch.tensor(images, dtype=torch.long)
        self.subjects = torch.tensor(subjects, dtype=torch.long)
        self.captions = captions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_data[idx],
            'label': self.labels[idx],
            'image': self.images[idx],
            'subject': self.subjects[idx],
            'caption': self.captions[idx]
        }

def extract_embeddings(model, dataloader):
    all_text_embeds = []
    all_text_labels = []

    model.eval()
    with torch.no_grad():
        for eeg_batch, text_batch in tqdm(dataloader, desc="Extracting test embeddings"):
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            text_emb = clip_model.encode_text(text_tokens).float()
            text_emb = F.normalize(text_emb, dim=-1)

            all_text_embeds.append(text_emb.cpu())
            all_text_labels.extend(text_batch)

    all_text_embeds = torch.cat(all_text_embeds, dim=0) # [N, 768]

    return all_text_embeds, all_text_labels

def retrieve_top_k_from_test_eeg(eeg_tensor,model,all_text_embeds,all_text_labels,device,k=5):
    model.eval()
    # eeg_tensor = eeg_tensor.unsqueeze(0).to(device)  # [1, n_chans, n_times]
    with torch.no_grad():
        emb = model(eeg_tensor)  # [1, 768]
        emb = F.normalize(emb, dim=-1)

    # emb = emb.cpu()  # if all_text_embeds is on CPU
    sim = emb @ all_text_embeds.t()  # [1, N]
    sim = sim.squeeze(0)            # [N]

    sorted_idx = torch.argsort(sim, descending=True)
    top_indices = sorted_idx[:k]
    results = []
    for idx_ in top_indices:
        label = all_text_labels[idx_]
        score = sim[idx_].item()
        results.append((label, score))
    return results

# %%
# Project unique captions into clip space
retrieval_df = test_df.drop_duplicates(subset=["captions"])
retrieval_dataset = EEGTextDataset(retrieval_df)
retrieval_dataloader  = DataLoader(retrieval_dataset, batch_size=128, shuffle=False)
unique_caption_embeds, unique_captions = extract_embeddings(cnt_model, retrieval_dataloader)

# Evaluate complete test test for caption retrieval given EEG
test_dataset = EEGTextDataset(test_df)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

all_eeg_embeds = []
all_true_captions = []

with torch.no_grad():
    for eeg_batch, caption_batch in tqdm(test_loader, desc="Extracting test EEG"):
        # EEG -> embeddings
        eeg_batch = eeg_batch.to(device)
        eeg_embeds = cnt_model(eeg_batch)        # [B, 768]
        eeg_embeds = F.normalize(eeg_embeds, dim=-1)
        all_eeg_embeds.append(eeg_embeds.cpu())
        # Store ground-truth caption for each sample (strings)
        all_true_captions.extend(caption_batch)

all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)  # [N, 768]
N = all_eeg_embeds.size(0)

print(f"Test EEG embedding shape: {all_eeg_embeds.shape}")
print(f"Number of test samples:   {N}")
print(f"Found {len(unique_captions)} unique test captions.")

all_eeg_embeds = all_eeg_embeds.to(device)
unique_caption_embeds = unique_caption_embeds.to(device)

sim_e2t = all_eeg_embeds @ unique_caption_embeds.t()  # shape [N, M]

caption_to_idx = {cap: idx for idx, cap in enumerate(unique_captions)}

true_text_indices = []
for cap in all_true_captions:
    correct_idx = caption_to_idx[cap]
    true_text_indices.append(correct_idx)
true_text_indices = torch.tensor(true_text_indices, device=device)

topks = [1, 5, 10]
hits_e2t = {k: 0 for k in topks}

for i in range(N):
    row = sim_e2t[i]  # [M]
    sorted_idx = torch.argsort(row, descending=True)
    correct_idx = true_text_indices[i]
    rank = (sorted_idx == correct_idx).nonzero(as_tuple=True)[0].item()
    for k in topks:
        if rank < k:
            hits_e2t[k] += 1

print("\n===== RETRIEVAL EVALUATION (EEG->Text) =====")
for k in topks:
    recall = hits_e2t[k] / N * 100
    print(f"Recall@{k}: {recall:.2f}%")

# %% [markdown]
# # Stable Diffusion with Beta Prior Semantic Interpolation

# %%
# test_df=test_df[test_df["subject"].isin([4])]

# indices = [901,1336,838,604,193,40]#,102,749,964,467,435,32,801,858,53,745,219,981]
# test_df=test_df.loc[indices]
# test_df = test_df[test_df['image'].isin(indices)]

# 20ms - 460ms as per the paper
tl=20
th=460

eeg_test = np.array(test_df['eeg'].apply(lambda x: x[:, tl:th]).tolist())
test_labels = test_df['label'].tolist()
test_images = test_df['image'].tolist()
test_subjects = test_df['subject'].tolist()
test_captions = test_df['captions'].tolist()

print("eeg_test.shape: ", eeg_test.shape)

# Create the dataset and DataLoader
batch_size = 1
test_dataset = EEGDataset(eeg_test, test_labels, test_images, test_subjects, test_captions)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", variant="fp16", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

vae = vae.to(device)
unet = unet.to(device)
text_encoder = text_encoder.to(device)

# %%
def custom_sd_pipe(class_embedding, caption_embedding):
    # Interpolated Conditional Text Embeddings
    interp_coef = beta_dist.sample()
    mean_text_embeddings = interp_coef * class_embedding + (1 - interp_coef) * caption_embedding
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    cond_embeddings = torch.cat([uncond_embeddings, mean_text_embeddings])
    cond_embeddings = cond_embeddings.half() # fp16 half precision is good enough for inference

    # Latent noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, 64, 64),
        generator=cuda_generator,
        device=device,
        dtype=torch.float16
    )
    latents = latents * scheduler.init_noise_sigma

    # Denoising
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=cond_embeddings).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decoding with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        img = vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (img * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]

    return pil_images[0]

# %%
from IPython.display import display

if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
    print(f"Created directory {WORK_DIR}")
if not os.path.exists(f"{WORK_DIR}/picture-gene"):
    os.makedirs(f"{WORK_DIR}/picture-gene")
    print(f"Created directory {WORK_DIR}/picture-gene")
if not os.path.exists(f"{WORK_DIR}/picture-gene-onlygt"):
    os.makedirs(f"{WORK_DIR}/picture-gene-onlygt")
    print(f"Created directory {WORK_DIR}/picture-gene-onlygt")

num_samples = 4
num_inference_steps = 100
guidance_scale = 7.5

alpha = 10
beta = 10
beta_dist = torch.distributions.Beta(alpha, beta)

track = {} # {"picture-gene-name": "retrieved caption"}
track_file = f"{WORK_DIR}/sd_track.pkl"

model.eval()
with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Testing", unit="batch"):
        eeg_test, label_test, image, subject, caption = test_data['eeg'], test_data['label'], test_data['image'], test_data['subject'], test_data['caption']
        eeg_test, label_test, image, subject = eeg_test.to(device), label_test.to(device), image.tolist(), subject.tolist()

        # Classfication
        outputs_test, _ = model(eeg_test)
        prediction = outputs_test.argmax(dim=-1).cpu().numpy()
        prompt = [f"{label_to_class[prediction[0]]}"]

        # Caption retrieval
        top_k_results = retrieve_top_k_from_test_eeg(
            eeg_tensor=eeg_test,
            model=cnt_model,
            all_text_embeds=unique_caption_embeds,
            all_text_labels=unique_captions,
            device=device,
            k=10
        )
        # print("GT Class: ", label_to_class[label_test.cpu().item()])
        # print("GT Caption: ", caption)
        # print("Predicted Class: ", prompt)

        # Text embeddings
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        class_embeds = text_encoder(text_input.input_ids.to(device))[0]
        text_input2 = tokenizer([top_k_result[0] for top_k_result in top_k_results], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        caption_embeds = text_encoder(text_input2.input_ids.to(device))[0]

        # Re-ranking
        query = class_embeds.view(1, -1)
        key = caption_embeds.view(10, -1)
        sim = F.cosine_similarity(query, key, dim=1)
        top_k = torch.topk(sim, k=num_samples, largest=True).indices.tolist()

        # copy original image to picture-gene-onlygt
        img=image[0]
        gt_path=image_to_path[img]
        gt_name=f"{img:04d}_{image_to_simple_class[img]}"
        source=f"{DATA_DIR}/imageNet_images/{gt_path}"
        destination=f"{WORK_DIR}/picture-gene-onlygt/{gt_name}.jpg" # 1995_shoe.jpg
        if not os.path.exists(destination):
            shutil.copyfile(source, destination)

        for i in range(num_samples):
            gene_image_name = f"{gt_name}_s{subject[0]}_{label_to_simple_class[prediction[0]]}_{i}.png" # 1995_shoe_s4_cat_0.png

            if os.path.exists(f"{WORK_DIR}/picture-gene/{gene_image_name}"):
                print(f"Skipping {gene_image_name}")
            else:
                generated_image = custom_sd_pipe(class_embeds, caption_embeds[top_k[i]])
                generated_image.save(f"{WORK_DIR}/picture-gene/{gene_image_name}")

                # print("Retrieved Caption: ", top_k_results[top_k[i]][0])
                # to track which retrieved caption resulted in this image
                track[gene_image_name]=top_k_results[top_k[i]][0]
                with open(track_file + '.tmp', 'wb') as temp_file:
                    pickle.dump(track, temp_file)
                os.rename(track_file + '.tmp', track_file)

                # display(generated_image)


# %%
!ls -la /workspace/work/picture-gene | wc -l

# %%
def cleanup():
    if os.path.exists(WORK_DIR):
        print(f"Deleting directory and contents: {WORK_DIR}")
        shutil.rmtree(WORK_DIR)

# cleanup()
