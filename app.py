from flask import Flask, render_template, request, jsonify
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# ==============================
# FAQ FILE LOAD
# ==============================

with open("faq.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)


questions = []
answers = []

for faq in faqs:
    question = faq.get("question")
    answer = faq.get("answer")

    if question and answer:
        questions.append(question)
        answers.append(answer)


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):

    text = text.lower().strip()

    replacements = {
        "b.tech": "btech",
        "b tech": "btech",
        "b. tech": "btech",
        "m.tech": "mtech",
        "m tech": "mtech",
        "m. tech": "mtech",
        "cse": "computer science engineering",
        "ai ml": "artificial intelligence machine learning"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================
# PREPARE QUESTIONS
# ==============================

cleaned_questions = [
    clean_text(q) for q in questions
]


# ==============================
# SMART TF-IDF MODEL
# ==============================

vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5)
)

question_vectors = vectorizer.fit_transform(
    cleaned_questions
)

# ==============================
# GET ANSWER
# ==============================

def get_answer(user_question):

    user_question = clean_text(user_question)

    if not user_question:
        return "Please apna question type karein."

    # Exact match
    if user_question in cleaned_questions:

        index = cleaned_questions.index(user_question)

        return answers[index]

    # Smart similarity matching
    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_index = similarity[0].argmax()
    best_score = similarity[0][best_index]

    # Strong match
    if best_score >= 0.40:
        return answers[best_index]

    # Related questions
    if best_score >= 0.18:

        top_indices = similarity[0].argsort()[::-1][:3]

        suggestions = []

        for index in top_indices:

            if similarity[0][index] >= 0.18:
                suggestions.append(questions[index])

        if suggestions:

            result = (
                "Mujhe exact answer nahi mila. "
                "Kya aap ye puchna chahte hain?\n\n"
            )

            for question in suggestions:
                result += "• " + question + "\n"

            return result

    # No match
    return (
        "Sorry, mujhe is question ka answer nahi mila. 😅\n\n"
        "Aap BRCM ke courses, admission, fees, hostel, "
        "library, placement, timing ya contact ke baare me puch sakte hain."
    )


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================
# ASK ROUTE
# ==============================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    user_question = data.get(
        "question",
        ""
    )

    answer = get_answer(
        user_question
    )

    return jsonify({
        "answer": answer
    })


# ==============================
# START FLASK
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )