import os
import uuid
import fitz
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from groq import Groq

# =========================
# Device Management
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# =========================
# Qdrant Setup
# =========================
collection_name = "rag_collection"
if "qdrant_client" not in st.session_state:
    client = QdrantClient(":memory:")
    st.session_state["qdrant_client"] = client
    existing = client.get_collections().collections
    names = [c.name for c in existing if c]
    if collection_name not in names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
qdrant_client = st.session_state["qdrant_client"]

# =========================
# PDF Reader
# =========================
def read_pdf(path: str) -> str:
    with open(path, 'rb') as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)

# =========================
# Text Chunker
# =========================
def chunk_text(text: str, size: int = 500, overlap: int = 100):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# =========================
# Ingest and Embed
# =========================
def process_and_embed(pdf_path: str) -> str:
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    text = read_pdf(pdf_path)
    passages = chunk_text(text)

    vectors = embedder.encode(passages, show_progress_bar=False, convert_to_tensor=True)
    if hasattr(vectors, 'cpu'):
        vectors = vectors.cpu().numpy()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": passage, "doc_id": doc_id}
        )
        for passage, vec in zip(passages, vectors)
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)
    return doc_id

# =========================
# Retrieve Context
# =========================
def retrieve_context(query: str, top_k: int = 3) -> str:
    q_vec = embedder.encode([query], convert_to_tensor=True)
    if hasattr(q_vec, 'cpu'):
        q_vec = q_vec.cpu().numpy()

    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=q_vec[0].tolist(),
        limit=top_k
    )
    return "\n\n".join(hit.payload["text"] for hit in hits)

# =========================
# Chatbot with Groq
# =========================
class Chatbot:
    def __init__(self, model: str):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY environment variable.")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.history = []

    def add(self, role: str, msg: str):
        self.history.append({"role": role, "content": msg})

    def chat(self, user_msg: str) -> str:
        ctx = retrieve_context(user_msg)
        system_msg = (
            "Use only the following context without exposing it:\n\n" + ctx
        )
        self.add("system", system_msg)
        self.add("user", user_msg)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=0.7
        )
        answer = resp.choices[0].message.content
        self.add("assistant", answer)
        return answer

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="RAG Chat with PDF", layout="wide")

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .stTextInput > div > div > input {
            width: 100%;
        }
        .stFileUploader {
            width: 100%;
        }
        .chat-bubble {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        .chat-user {
            background-color: #d2e3fc;
        }
        .chat-assistant {
            background-color: #e9ecef;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🧠 RAG Chat with PDF")

    models = ["qwen-qwq-32b", "deepseek-r1-distill-llama-70b"]
    chosen = st.selectbox("Choose model:", models)

    if "bot" not in st.session_state or st.session_state["bot"].model != chosen:
        st.session_state["bot"] = Chatbot(chosen)
    bot = st.session_state["bot"]

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        tmp = f"/tmp/{uploaded.name}"
        with open(tmp, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved to {tmp}")
        if st.button("Ingest"):
            did = process_and_embed(tmp)
            st.success(f"Ingested {did}")

    st.subheader("Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    query = st.text_input("Your question:")
    if st.button("Send") and query:
        reply = bot.chat(query)
        st.session_state["chat_history"].append(("You", query))
        st.session_state["chat_history"].append(("Assistant", reply))

    st.subheader("Chat History")
    for user, msg in st.session_state["chat_history"]:
        if user == "You":
            st.markdown(
                f'<div class="chat-bubble chat-user"><b>{user}:</b> {msg}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble chat-assistant"><b>{user}:</b> {msg}</div>',
                unsafe_allow_html=True
            )

    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []


if __name__ == "__main__":
    main()
