# app.py
import os
import toml
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
import streamlit as st
import json
import tempfile
from google.oauth2 import service_account
from vertexai import init


# 1. Extract from Streamlit secrets
project_id = st.secrets["general"]["project"]
LOCATION = "us-central1"
adc_info = st.secrets["google_credentials"]
json_str = json.dumps(adc_info)
# 3. Load credentials from the temp JSON file
creds = service_account.Credentials.from_service_account_file(json_str)

# 4. Init Vertex AI
init(project=project_id, location=LOCATION, credentials=creds)


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
st.set_page_config(page_title="🔍 Multimodal Search")
st.title("S6 Search term based image search")

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
