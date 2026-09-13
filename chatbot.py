from flask import Flask, request, jsonify, render_template
import json
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# =========================
# Load FAQ Data
# =========================

with open("faq.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)


# Questions and Answers
questions = []
answers = []

for faq in faqs:
    question = faq.get("question") or faq.get("Question")
    answer = faq.get("answer") or faq.get("Answer")

    if question and answer:
        questions.append(question)
        answers.append(answer)


# =========================
# AI Search Setup
# =========================

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


# =========================
# Chatbot Function
# =========================

def get_answer(user_question):

    user_question = user_question.lower().strip()

    # Common spelling / wording normalization
    replacements = {
        "b tech": "btech",
        "b.tech": "btech",
        "b. tech": "btech",
        "ai ml": "aiml",
        "ai & ml": "aiml",
        "fees": "fee",
        "admission process": "admission",
        "college location": "college kaha hai"
    }

    for old, new in replacements.items():
        user_question = user_question.replace(old, new)

    # TF-IDF matching
    question_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        question_vector,
        question_vectors
    )

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    # Good match
    if best_score >= 0.45:
        return answers[best_match_index]

    # Fallback: find similar questions
    all_questions = questions

    similar_questions = difflib.get_close_matches(
        user_question,
        all_questions,
        n=3,
        cutoff=0.15
    )

    if similar_questions:

        suggestions = "\n".join(
            [f"• {q}" for q in similar_questions]
        )

        return (
            "🤔 Mujhe exact answer nahi mila.\n\n"
            "Kya aap inme se koi question poochna chahte hain?\n\n"
            + suggestions
        )

    return (
        "🤔 Sorry, mujhe iska answer nahi mila.\n\n"
        "Aap BRCM College ke courses, admission, fees, "
        "hostel, library, placement ya contact details ke baare me pooch sakte hain."
    )
# =========================
# Home Page
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# Ask Chatbot
# =========================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    user_question = data.get("question", "")

    answer = get_answer(user_question)

    return jsonify({
        "answer": answer
    })


# =========================
# Run Server
# =========================

if __name__ == "__main__":
    app.run(debug=True)