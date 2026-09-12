"""Web interface for the FAQ chatbot.

Run with:  python app.py
Then open http://127.0.0.1:5000

This is a thin layer: Flask serves one page and one JSON endpoint, and all the
matching logic stays in chatbot.py. The page talks to /api/ask with fetch(), so
only the reply is sent over the wire and the browser keeps its own state - the
chat never re-renders from scratch between messages.
"""

from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)

# Built once at import, not per request. Re-reading the JSON and re-fitting the
# vectorizer on every message would add pointless latency to every reply.
bot = FAQChatbot()


@app.route("/")
def index():
    return render_template(
        "index.html",
        questions=bot.questions,
        faq_count=len(bot.faqs),
        threshold=bot.threshold,
    )


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", ""))

    response = bot.ask(question)

    return jsonify(
        {
            "text": response.text,
            "confident": response.confident,
            "score": round(response.score, 3),
            "matched_question": response.matched_question,
            "suggestions": [m.question for m in (response.suggestions or [])],
        }
    )


if __name__ == "__main__":
    print(f"FAQ chatbot ready - {len(bot.faqs)} questions loaded.")
    print("Open http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
