import chromadb
import torch
import open_clip

# -------------------------
# Load CLIP
# -------------------------

model, _, _ = open_clip.create_model_and_transforms(
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

collection = client.get_collection(
    "multimodal_images"
)

# -------------------------
# Query
# -------------------------

query = input(
    "\nEnter search query: "
)

# -------------------------
# Text Embedding
# -------------------------

tokens = tokenizer(
    [query]
)

with torch.no_grad():

    embedding = model.encode_text(
        tokens
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

# -------------------------
# Search
# -------------------------

results = collection.query(
    query_embeddings=[vector],
    n_results=3
)

print(
    "\n===== RESULTS =====\n"
)

for item in results["documents"][0]:

    print(item)