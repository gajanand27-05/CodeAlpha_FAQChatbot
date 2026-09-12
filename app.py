"""Streamlit chat interface for the FAQ chatbot.

Run with:  streamlit run app.py

All the matching logic lives in chatbot.py - this file only handles display and
the conversation history.
"""

import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="AI/ML FAQ Chatbot", page_icon="🤖")


@st.cache_resource
def load_bot() -> FAQChatbot:
    """Build the bot once and reuse it.

    Without the cache decorator Streamlit would re-read the JSON and re-fit the
    vectorizer on every single interaction, since it reruns the whole script
    each time the user types.
    """
    return FAQChatbot()


bot = load_bot()

st.title("🤖 AI/ML FAQ Chatbot")
st.caption(
    f"Ask about machine learning concepts. {len(bot.faqs)} questions in the dataset. "
    "Matching is TF-IDF + cosine similarity - no LLM involved."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. **Preprocess** - lowercase, strip punctuation, drop stopwords, lemmatize
2. **Vectorise** - TF-IDF turns the text into numbers
3. **Compare** - cosine similarity against every stored question
4. **Respond** - return the best match, or admit it doesn't know
        """
    )
    show_scores = st.checkbox("Show match scores", value=False)
    st.metric("Confidence threshold", f"{bot.threshold:.2f}")
    st.caption("Below this score the bot says it doesn't know rather than guessing.")

    with st.expander("Questions I can answer"):
        for question in bot.questions:
            st.write(f"- {question}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. Ask me anything about AI or machine learning."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])

if prompt := st.chat_input("What is overfitting?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = bot.ask(prompt)

    caption = None
    if show_scores:
        caption = f"score {response.score:.3f}"
        if response.matched_question:
            caption += f" · matched: “{response.matched_question}”"

    with st.chat_message("assistant"):
        st.write(response.text)
        if caption:
            st.caption(caption)
        if response.suggestions:
            st.write("Did you mean one of these?")
            for match in response.suggestions:
                st.write(f"- {match.question}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response.text, "caption": caption}
    )
