import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

# === CONFIG ===
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
INDEX_NAME = "property-assistant"

# === INIT ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = OpenAI(api_key=OPENAI_API_KEY)

# === UI ===
st.set_page_config(page_title="Property Advisor", layout="centered")
st.title("🏡 Property Advisor in Your Pocket")
st.write("Ask any UK property question and get smart advice + real video clips.")

question = st.text_input("💬 What would you like to know?", placeholder="e.g. What should I ask an estate agent?")

# === SEARCH + GPT ===
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

            text = meta.get("text", "")
            url = meta.get("youtube_url", "#")

            responses.append((title, text, url))
            if len(responses) == 3:
                break

    if responses:
        # Build prompt context for GPT
        context = "\n\n".join([f"{t}\n{text}" for t, text, _ in responses])
        prompt = f"""You are a helpful UK property advisor.

The user asked: "{question}"

Below are short transcripts from expert videos. First, give a clear and helpful answer in a friendly tone. Do NOT insert YouTube links. We'll show those separately.

{context}
"""

        with st.spinner("🤖 Generating expert answer..."):
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful UK property advisor using expert video clips."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

        # === OUTPUT ===
        st.success("✅ Here's your answer:")
        st.markdown(gpt_response.choices[0].message.content)

        # === REAL LINKS ===
        st.markdown("### 📺 Watch the related clips:")
        for i, (_, _, url) in enumerate(responses, 1):
            st.markdown(f"{i}. [Watch on YouTube]({url})", unsafe_allow_html=True)

    else:
        st.warning("No matching clips found. Try rephrasing your question.")
