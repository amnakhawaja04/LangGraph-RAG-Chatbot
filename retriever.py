# retriever.py
import os
import pickle
import faiss
from glob import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = r"C:\Users\afarooq\Downloads\lmkr_graph\lmkr_data"

ARTIFACT_DIR = "./artifacts"
FAISS_PATH = f"{ARTIFACT_DIR}/lmkr_faiss.index"
DOCS_PATH = f"{ARTIFACT_DIR}/lmkr_docs.pkl"

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_raw_docs():
    paths = sorted(glob(f"{DATA_DIR}/*.txt"))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(p)}))
    return docs


def build_faiss():
    print("Building FAISS...")

    raw_docs = load_raw_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    embeddings = embed_model.encode([d.page_content for d in chunks], show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS saved.")
    return index, chunks


def load_faiss():
    if os.path.exists(FAISS_PATH) and os.path.exists(DOCS_PATH):
        print("Loading FAISS...")
        index = faiss.read_index(FAISS_PATH)
        with open(DOCS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return build_faiss()


def faiss_search(query, index, chunks, k=4):
    q_emb = embed_model.encode([query])
    distances, idxs = index.search(q_emb, k)

    results = []
    for i in idxs[0]:
        if i != -1:
            results.append(chunks[i])
    return results
