from retriever import LegalRetriever
from augmentor import LegalAugmentor
from generator import LegalGenerator
from rag_pipeline import LegalRAGPipeline


def main():

    print("🤖 جاري تحميل نظام المساعد القانوني...")

    # 1. Retrieval
    retriever = LegalRetriever()

    # 2. Augmentation
    augmentor = LegalAugmentor()

    # 3. Generation using Qwen
    generator = LegalGenerator()

    # 4. Build complete RAG pipeline
    rag_pipeline = LegalRAGPipeline(
        retriever=retriever,
        augmentor=augmentor,
        generator=generator
    )

    print("✅ النظام جاهز!")

    while True:

        query = input(
            "\nادخل سؤالك القانوني "
            "(أو اكتب 'exit' للخروج): "
        )

        if query.lower() == "exit":
            print("👋 تم إنهاء البرنامج.")
            break

        if not query.strip():
            continue

        answer, docs = rag_pipeline.generate_answer(
            query,
            top_k=3
        )

        print("\n--- الإجابة القانونية ---")
        print(answer)


if __name__ == "__main__":
    main()