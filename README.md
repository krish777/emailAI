# emailAI
📧 Email Assistant – AI Powered Email Summarizer, Sentiment Analyzer & Auto-Replier
This project is a lightweight Streamlit web app that uses Hugging Face Transformers and TextBlob to:

✅ Summarize email content

✅ Detect the sentiment of the email (Positive / Negative / Neutral)

✅ Generate a contextual auto-reply using an AI model

🛠️ Built With
Streamlit – Python UI framework

Hugging Face Transformers – NLP models

TextBlob – Sentiment analysis

PyTorch – ML backend

# Customize Models
This project uses two Hugging Face models by default:

# Summarization model: facebook/bart-large-cnn

Auto-reply generation model: microsoft/DialoGPT-medium

If you'd like to change these models, you can update the model names in the email.py file:

# 1. For summarization:

Find this line:

python
Copy
Edit
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
You can replace "facebook/bart-large-cnn" with another summarization model, such as:

"sshleifer/distilbart-cnn-12-6"

"t5-base"

"google/pegasus-xsum"

Browse more at: https://huggingface.co/models?pipeline_tag=summarization

# 2. For auto-reply generation:

Find these lines:

python
Copy
Edit
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
You can replace "microsoft/DialoGPT-medium" with other conversational models, such as:

"microsoft/DialoGPT-large"

"microsoft/DialoGPT-small"

"gpt2"

Browse more at: https://huggingface.co/models?pipeline_tag=text-generation

⚠️ Make sure the new models are compatible with the pipeline task (summarization or text-generation) and install any additional dependencies if needed.

# 3. Sentiment Analysis
Uses TextBlob, a lightweight NLP library.
Sentiment is calculated with:

python
Copy
Edit
blob = TextBlob(email_text)
sentiment = blob.sentiment.polarity
