import json
import re
from pathlib import Path

from rouge_score import rouge_scorer


# =========================================================
# Files
# =========================================================

BASE_DIR = Path(__file__).parent

QUESTIONS_FILE = BASE_DIR / "test_questions.json"
RESULTS_FILE = BASE_DIR / "rag_results.json"
REPORT_FILE = BASE_DIR / "evaluation_report.json"


# =========================================================
# Load JSON Files
# =========================================================

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Arabic Text Normalization
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).strip().lower()

    # Remove Arabic diacritics
    arabic_diacritics = "ًٌٍَُِّْـ"

    for char in arabic_diacritics:
        text = text.replace(char, "")

    # Normalize Arabic letters
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize punctuation
    text = re.sub(
        r"[،,؛;:!?؟.!]",
        " ",
        text
    )

    # Remove brackets
    text = re.sub(
        r"[\(\)\[\]\{\}]",
        " ",
        text
    )

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# =========================================================
# Exact Match
# =========================================================

def exact_match(predicted_answer, expected_answer):

    predicted = normalize_text(predicted_answer)
    expected = normalize_text(expected_answer)

    return predicted == expected


# =========================================================
# ROUGE
# =========================================================

def calculate_rouge(predicted_answer, expected_answer):

    predicted = normalize_text(predicted_answer)
    expected = normalize_text(expected_answer)

    if not predicted or not expected:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False
    )

    scores = scorer.score(
        expected,
        predicted
    )

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


# =========================================================
# Keyword Match
# =========================================================

def keyword_match(predicted_answer, expected_answer):

    predicted = normalize_text(predicted_answer)
    expected = normalize_text(expected_answer)

    if not predicted or not expected:
        return False

    expected_words = expected.split()
    predicted_words = set(predicted.split())

    # Remove very common words
    stop_words = {
        "من",
        "في",
        "و",
        "او",
        "او",
        "على",
        "الى",
        "عن",
        "ان",
        "أن",
        "هو",
        "هي",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "لا",
        "ما",
        "يكون",
        "تكون",
        "مع",
        "بين",
        "كل",
        "لهم",
        "له",
        "لها"
    }

    important_words = [
        word
        for word in expected_words
        if len(word) >= 3 and word not in stop_words
    ]

    if not important_words:
        return False

    matched_words = sum(
        1
        for word in important_words
        if word in predicted_words
    )

    match_ratio = matched_words / len(important_words)

    # If most important words exist in the prediction
    return match_ratio >= 0.50


# =========================================================
# Final Answer Evaluation
# =========================================================

def evaluate_answer(predicted_answer, expected_answer):

    exact = exact_match(
        predicted_answer,
        expected_answer
    )

    rouge = calculate_rouge(
        predicted_answer,
        expected_answer
    )

    keyword = keyword_match(
        predicted_answer,
        expected_answer
    )

    # Final correctness
    #
    # Exact Match = correct
    # OR
    # Keyword Match = correct
    #
    # This handles different Arabic answer wording.

    correct = exact or keyword

    return {
        "correct": correct,
        "exact_match": exact,
        "keyword_match": keyword,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"]
    }


# =========================================================
# Evaluate All Questions
# =========================================================

def evaluate_all(rag_results):

    questions = load_questions()

    predicted_answers = {
        item["id"]: item.get(
            "predicted_answer",
            ""
        )
        for item in rag_results
    }

    results = []

    for question in questions:

        question_id = question["id"]

        expected = question["expected_answer"]

        predicted = predicted_answers.get(
            question_id,
            ""
        )

        evaluation = evaluate_answer(
            predicted,
            expected
        )

        results.append({

            "id": question_id,

            "question": question["question"],

            "type": question["type"],

            "expected_answer": expected,

            "predicted_answer": predicted,

            "correct": evaluation["correct"],

            "exact_match": evaluation["exact_match"],

            "keyword_match": evaluation["keyword_match"],

            "rouge1": evaluation["rouge1"],

            "rouge2": evaluation["rouge2"],

            "rougeL": evaluation["rougeL"]
        })

    return results


# =========================================================
# Calculate Metrics
# =========================================================

def calculate_metrics(results):

    total = len(results)

    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    correct = sum(
        1
        for result in results
        if result["correct"]
    )

    accuracy = correct / total

    # -----------------------------------------------------
    # Precision / Recall / F1
    #
    # Simple evaluation based on correct answers.
    # -----------------------------------------------------

    true_positive = correct

    false_positive = total - correct

    false_negative = total - correct

    if true_positive + false_positive > 0:

        precision = (
            true_positive /
            (true_positive + false_positive)
        )

    else:

        precision = 0.0

    if true_positive + false_negative > 0:

        recall = (
            true_positive /
            (true_positive + false_negative)
        )

    else:

        recall = 0.0

    if precision + recall > 0:

        f1_score = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1_score = 0.0

    # -----------------------------------------------------
    # Average ROUGE
    # -----------------------------------------------------

    rouge1 = sum(
        result["rouge1"]
        for result in results
    ) / total

    rouge2 = sum(
        result["rouge2"]
        for result in results
    ) / total

    rougeL = sum(
        result["rougeL"]
        for result in results
    ) / total

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1_score,

        "rouge1": rouge1,

        "rouge2": rouge2,

        "rougeL": rougeL
    }


# =========================================================
# Save Evaluation Report
# =========================================================

def save_report(results, metrics):

    report = {

        "metrics": metrics,

        "results": results
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=4
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    questions = load_questions()

    rag_results = load_results()

    print(
        f"Loaded {len(questions)} test questions."
    )

    print(
        f"Loaded {len(rag_results)} RAG results."
    )

    results = evaluate_all(
        rag_results
    )

    # -----------------------------------------------------
    # Individual Results
    # -----------------------------------------------------

    print("\nEvaluation Results:")

    for result in results:

        print(
            f"Question {result['id']}: "
            f"{result['correct']} "
            f"| Exact Match: "
            f"{result['exact_match']} "
            f"| Keyword Match: "
            f"{result['keyword_match']} "
            f"| ROUGE-1: "
            f"{result['rouge1']:.2%} "
            f"| ROUGE-2: "
            f"{result['rouge2']:.2%} "
            f"| ROUGE-L: "
            f"{result['rougeL']:.2%}"
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    print("\nMetrics:")

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"F1-Score: "
        f"{metrics['f1_score']:.2%}"
    )

    print(
        f"ROUGE-1: "
        f"{metrics['rouge1']:.2%}"
    )

    print(
        f"ROUGE-2: "
        f"{metrics['rouge2']:.2%}"
    )

    print(
        f"ROUGE-L: "
        f"{metrics['rougeL']:.2%}"
    )

    # -----------------------------------------------------
    # Save Report
    # -----------------------------------------------------

    save_report(
        results,
        metrics
    )

    print(
        f"\nReport saved to: "
        f"{REPORT_FILE}"
    )