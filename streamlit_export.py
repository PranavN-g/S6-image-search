# app.py
import os
import json
import requests
import faiss
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
from google.oauth2 import service_account
from vertexai import init
from vertexai.vision_models import MultiModalEmbeddingModel

# 1) GCP & Vertex AI setup via Streamlit Secrets
creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
creds = service_account.Credentials.from_service_account_info(creds_info)
PROJECT = creds_info["project_id"]
LOCATION = "us-central1"
init(project=PROJECT, location=LOCATION, credentials=creds)

# 2) Load Vertex AI model
try:
    mmodel = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
except Exception as e:
    st.error(f"Failed to load Vertex AI model: {e}")
    st.stop()

# 3) Load precomputed FAISS index & metadata
INDEX_PATH = "sku_index.faiss"
META_PATH  = "metadata.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)  # list of [SKU_ID, URL]

# 4) Streamlit UI
st.set_page_config(page_title="🔍 Multimodal Search", layout="wide")
st.title("🔍 Click-and-Search")

query = st.text_input("Enter search text:")
if query:
    with st.spinner("Searching…"):
        resp = mmodel.get_embeddings(contextual_text=query)
        emb = np.array(resp.text_embedding, dtype="float32")[None]
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, 6)

    cols = st.columns(3)
    for (idx, score), col in zip(zip(indices[0], scores[0]), cols * 2):
        sku, url = metadata[idx]
        try:
            img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
            col.image(img, caption=f"{sku} • {score:.3f}", use_container_width=True)
        except Exception:
            col.write(f"{sku} • {score:.3f}\n(image load failed)")
