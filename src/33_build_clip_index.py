import os

import chromadb
import torch
import open_clip

from PIL import Image

# -------------------------
# Load CLIP
# -------------------------

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer(
    "ViT-B-32"
)

# -------------------------
# Chroma
# -------------------------

client = chromadb.PersistentClient(
    path="clip_chroma_db"
)

collection = client.get_or_create_collection(
    name="multimodal_images"
)

# -------------------------
# Image Folder
# -------------------------

IMAGE_DIR = "images"

# -------------------------
# Process Images
# -------------------------

for filename in os.listdir(
    IMAGE_DIR
):

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = preprocess(
        Image.open(image_path)
    ).unsqueeze(0)

    with torch.no_grad():

        embedding = model.encode_image(
            image
        )

        embedding = (
            embedding
            / embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

    vector = (
        embedding[0]
        .cpu()
        .numpy()
        .tolist()
    )

    collection.add(
        ids=[filename],
        embeddings=[vector],
        documents=[filename]
    )

    print(
        f"Indexed {filename}"
    )

print(
    "\nFinished indexing."
)