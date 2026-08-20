# CodeAlpha_FAQChatbot

**CodeAlpha Artificial Intelligence Internship — Task 2: Chatbot for FAQs**

An FAQ chatbot that answers questions about AI/ML concepts. You type a question in
your own words; the bot finds the closest matching question in its FAQ dataset and
replies with that answer.

> Status: in progress. This README is filled in as the project is built.

## What it does

- Stores a dataset of AI/ML question–answer pairs
- Cleans and tokenizes text using NLP preprocessing
- Turns text into numbers with TF-IDF, and finds the best match using cosine similarity
- Serves the answer through a chat interface built with Streamlit

## Tech stack

| Piece | Library |
|---|---|
| Text preprocessing | NLTK |
| Vectorising + similarity | scikit-learn |
| Chat interface | Streamlit |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Author

Built by **gajanand27-05** as part of the CodeAlpha AI Internship.
