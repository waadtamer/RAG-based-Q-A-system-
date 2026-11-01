import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# ====== الإعدادات ======
EMBED_MODEL = "all-MiniLM-L6-v2"   # غيّريه لنفس موديل الـ embeddings في النوتبوك
DB_PATH = r"C:\Users\waadt\Desktop\RAG2\chroma_db"
COLLECTION_NAME = "course_docs"
LLM_MODEL = "gemini-3.5-flash-lite"
TOP_K = 4
# =======================

st.set_page_config(page_title="Course Q&A RAG", page_icon="📚", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #000000; }

h1 {
    background: linear-gradient(90deg, #ff4b9a, #ffb347, #4be3ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stTextInput label p { color: #ffb347 !important; font-size: 18px; }
.stTextInput input { color: #4be3ff; background-color: #111; }

.stMarkdown p { color: #7CFFB2; font-size: 17px; }

[data-testid="stExpander"] summary p { color: #ff4b9a !important; }
[data-testid="stCaptionContainer"] { color: #c3a6ff !important; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Course Q&A RAG")


@st.cache_resource
def load_resources():
    model = SentenceTransformer(EMBED_MODEL)
    db = chromadb.PersistentClient(path=DB_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return model, collection, llm


model, collection, llm = load_resources()

if collection.count() == 0:
    st.warning("الـ collection فاضية. اتأكدي من DB_PATH واسم الـ collection.")
    st.stop()

question = st.text_input("اسألي سؤالك:")

if question:
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=TOP_K)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join(docs)

    prompt = f"""Answer the question based on the context below.
If the answer is not in the context, say you don't know.
Answer in the same language as the question.

Context:
{context}

Question: {question}
Answer:"""

    with st.spinner("بفكر..."):
        response = llm.models.generate_content(model=LLM_MODEL, contents=prompt)

    st.markdown(response.text)

    with st.expander("المصادر"):
        for doc, meta in zip(docs, metas):
            src = meta.get("source", "unknown") if meta else "unknown"
            page = meta.get("page", "") if meta else ""
            st.caption(f"📄 {src} {('- page ' + str(page)) if page != '' else ''}")
            st.caption(doc[:300] + "...")