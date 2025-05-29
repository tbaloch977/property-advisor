import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai

# === CONFIG ===
PINECONE_API_KEY = "pcsk_Z8vs3_GhRc642dA1H6jNoNLgWNqYdrjQjMJTnd1ibERHQkudAao6dvmQGzmDU3CWHs78a"  # Replace with your actual key
OPENAI_API_KEY = "openai"      # Replace with your actual key
INDEX_NAME = "property-assistant"

# === INIT ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# === UI ===
st.set_page_config(page_title="Property Advisor", layout="centered")
st.title("🏡 Property Advisor in Your Pocket")
st.write("Ask any UK property question and get smart advice + real clips.")

question = st.text_input("💬 What would you like to know?", placeholder="e.g. What should I ask an estate agent?")

# === Search & GPT ===
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
        context = "\n\n".join([f"{t}\n{text}" for t, text, _ in responses])
        prompt = f"""You are a helpful UK property advisor.
The user asked: "{question}"

Below are short transcripts from expert videos. First, give a clear answer. Then list 2–3 bullet points with real YouTube links to watch.

{context}
"""

        with st.spinner("🤖 Generating expert summary..."):
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful UK property advisor using expert video clips."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

        st.success("✅ Here's your answer:")
        st.markdown(gpt_response.choices[0].message.content)

    else:
        st.warning("No matching clips found. Try rephrasing your question.")
