import argparse
from pathlib import Path

from augmentor import LegalAugmentor
from document_processor import process_contract_dynamically
from generator import LegalGenerator
from rag_pipeline import LegalRAGPipeline
from retriever import LegalRetriever


def main():
    parser = argparse.ArgumentParser(description="Arabic Legal AI Assistant (CLI)")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="مسار ملف العقد (PDF / DOCX / TXT). إذا لم يتم تحديده، سيتم استخدام الفهرس المحفوظ مسبقاً.",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=3,
        help="عدد البنود المسترجعة",
    )
    args = parser.parse_args()

    print("🤖 جاري تحميل نظام المساعد القانوني العربي...")

    if args.file:
        contract_path = Path(args.file)
        if not contract_path.exists():
            print(f"❌ الملف غير موجود: {contract_path}")
            return
        print(f"📄 جاري استخراج وفهرسة العقد: {contract_path.name}...")
        retriever, chunks, pages = process_contract_dynamically(
            file_source=contract_path,
            filename=contract_path.name,
        )
        print(f"✅ تمت الفهرسة بنجاح! ({len(pages)} صفحة، {len(chunks)} مقطع/بند).")
    else:
        print("📁 استخدام الفهرس المحفوظ مسبقاً...")
        try:
            retriever = LegalRetriever()
        except Exception as e:
            print(f"⚠️ تعذر تحميل الفهرس المخزن: {e}")
            print("💡 نصيحة: مرر مسار العقد عبر --file مثل: python main.py --file my_contract.pdf")
            return

    augmentor = LegalAugmentor()
    generator = LegalGenerator()

    rag_pipeline = LegalRAGPipeline(
        retriever=retriever,
        augmentor=augmentor,
        generator=generator,
    )

    print("✅ المساعد القانوني جاهز لاستقبال الأسئلة!")

    while True:
        try:
            query = input(
                "\nادخل سؤالك القانوني حول العقد (أو اكتب 'exit' للخروج): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 تم إنهاء البرنامج.")
            break

        if query.lower() in ("exit", "quit", "خروج"):
            print("👋 تم إنهاء البرنامج.")
            break

        if not query:
            continue

        print("\n🔍 جاري البحث في بنود العقد واستنباط الإجابة...")
        answer, docs = rag_pipeline.generate_answer(query, top_k=args.top_k)

        print("\n" + "=" * 60)
        print("--- الإجابة القانونية ---")
        print("=" * 60)
        print(answer)

        print("\n--- البنود المسترجعة كدليل ---")
        for i, doc in enumerate(docs, 1):
            pages = doc.get("source_pages", [])
            score = doc.get("score", 0.0)
            print(f"[{i}] (الصفحات: {pages} | درجة التطابق: {score:.2%}):")
            print(doc.get("text", "")[:250] + "...")
            print("-" * 40)


if __name__ == "__main__":
    main()