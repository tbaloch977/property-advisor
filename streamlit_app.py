from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import streamlit as st

# INIT
pc = Pinecone(api_key="pcsk_Z8vs3_GhRc642dA1H6jNoNLgWNqYdrjQjMJTnd1ibERHQkudAao6dvmQGzmDU3CWHs78a")
index = pc.Index("property-assistant")

# ==== INIT ====
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index = pinecone.Index(INDEX_NAME)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==== UI ====
st.set_page_config(page_title="Property Advisor", layout="centered")
st.title("🏡 Property Advisor in Your Pocket")
st.write("Ask any property question and get real video advice with timestamped YouTube clips.")

# ==== INPUT ====
question = st.text_input("💬 What would you like to know?", placeholder="e.g. What should I ask an estate agent?")

# ==== SEARCH + DISPLAY ====
if question:
    with st.spinner("🔍 Searching expert clips..."):
        vector = embedder.encode(question).tolist()
        results = index.query(vector=vector, top_k=10, include_metadata=True)

        seen = set()
        responses = []

        for match in results["matches"]:
            meta = match["metadata"]
            title = meta.get("video_title", "Unknown")
            if title in seen:
                continue
            seen.add(title)

            summary = meta.get("text", "")[:200] + "..."
            url = meta.get("youtube_url", "#")

            responses.append((title, summary, url))
            if len(responses) == 3:
                break

    if responses:
        st.success("✅ Here's what we found:")
        for i, (title, summary, url) in enumerate(responses, 1):
            st.markdown(f"### {i}. {title}")
            st.write(summary)
            st.markdown(f"📺 [Watch the clip]({url})\n")
    else:
        st.warning("No matching clips found. Try rephrasing your question.")
