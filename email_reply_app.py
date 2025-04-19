import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob
import torch

# Load models once
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return summarizer, tokenizer, model

summarizer, tokenizer, model = load_models()

# Streamlit UI
st.title("📧 Email Assistant App")
st.write("This app summarizes, analyzes sentiment, and generates a reply to your email.")

email_text = st.text_area("Paste your email content below:", height=200)

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter email text.")
    else:
        # --- Summarization ---
        with st.spinner("Summarizing..."):
            summary = summarizer(email_text, max_length=50, min_length=25, do_sample=False)
            st.subheader("📌 Summary")
            st.success(summary[0]['summary_text'])

        # --- Sentiment ---
        with st.spinner("Analyzing sentiment..."):
            blob = TextBlob(email_text)
            sentiment = blob.sentiment.polarity
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            st.subheader("🧠 Sentiment")
            st.info(f"Sentiment: **{sentiment_label}** (score: {sentiment:.2f})")

        # --- Reply Generation ---
        with st.spinner("Generating reply..."):
            input_ids = tokenizer.encode(email_text + tokenizer.eos_token, return_tensors='pt')
            reply_ids = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            st.subheader("✉️ Suggested Reply")
            st.write(reply)