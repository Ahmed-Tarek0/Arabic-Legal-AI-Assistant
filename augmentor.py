from typing import List, Dict, Any


SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في القوانين واللوائح والعقود باللغة العربية.

مهمتك هي الإجابة على سؤال المستخدم اعتمادًا حصريًا على النصوص القانونية
المسترجعة والمقدمة لك في السياق.

القواعد:
1. استخدم المعلومات الموجودة في النصوص المسترجعة فقط.
2. لا تضف معلومات من معرفتك الخارجية.
3. لا تخترع مواد قانونية أو أرقام مواد أو أحكام.
4. إذا لم تكن الإجابة موجودة أو كانت المعلومات غير كافية، قل بوضوح:
"المعلومة غير متوفرة في المستندات المتاحة".
5. أجب باللغة العربية بأسلوب قانوني رسمي.
6. اذكر مصدر المعلومة أو رقم المادة إذا كان موجودًا في السياق.
"""


class LegalAugmentor:
    """Handles the Augmentation phase of the RAG pipeline."""

    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:

        context_blocks = []

        for idx, doc in enumerate(retrieved_docs, start=1):

            text = doc.get("text", doc.get("page_content", ""))

            source = doc.get(
                "source",
                doc.get("doc_id", f"المستند رقم {idx}")
            )

            article = doc.get("article", "")
            pages = doc.get("source_pages", "")

            metadata = [f"المصدر: {source}"]

            if article:
                metadata.append(f"المادة: {article}")

            if pages:
                metadata.append(f"الصفحات: {pages}")

            header = " | ".join(metadata)

            context_blocks.append(
                f"[المصدر {idx}]\n"
                f"{header}\n\n"
                f"{text}"
            )

        return "\n\n---\n\n".join(context_blocks)

    def augment(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:

        context = self.build_context(retrieved_docs)

        user_prompt = f"""النصوص القانونية المسترجعة:

{context}

سؤال المستخدم:
{query}

أجب على السؤال اعتمادًا على النصوص السابقة فقط."""

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]