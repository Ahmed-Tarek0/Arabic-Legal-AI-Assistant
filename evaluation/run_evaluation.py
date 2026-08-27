import sys
import json
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import LegalRAGPipeline
from retriever import LegalRetriever
from augmentor import LegalAugmentor
from generator import LegalGenerator


# =========================
# Files
# =========================

QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "rag_results.json"


# =========================
# Load questions
# =========================

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)


print(f"Loaded {len(questions)} test questions.")


# =========================
# Create RAG components
# =========================

retriever = LegalRetriever()
augmentor = LegalAugmentor()
generator = LegalGenerator()

pipeline = LegalRAGPipeline(
    retriever,
    augmentor,
    generator
)


# =========================
# Run questions
# =========================

results = []

for item in questions:

    question_id = item["id"]
    question = item["question"]

    print(f"\nRunning Question {question_id}:")
    print(question)

    try:

        answer, retrieved_docs = pipeline.generate_answer(
            question,
            top_k=3
        )

        print("Answer:")
        print(answer)

        results.append({
            "id": question_id,
            "question": question,
            "predicted_answer": answer
        })

    except Exception as e:

        print("Error:", e)

        results.append({
            "id": question_id,
            "question": question,
            "predicted_answer": ""
        })


# =========================
# Save results
# =========================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )


print("\n=========================")
print("RAG evaluation results saved.")
print(f"File: {RESULTS_FILE}")
print("=========================")