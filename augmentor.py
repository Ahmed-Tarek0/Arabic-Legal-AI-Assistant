"""
Augmentation module for Arabic Legal Assistant.
Builds concise, direct prompts focused on answering the user's question accurately.
"""

from typing import Any, Dict, List

SYSTEM_PROMPT = """أنت مستشار قانوني ذكي متخصص في قراءة وتحليل العقود باللغة العربية.

مهمتك:
الإجابة على سؤال المستخدم بشكل مباشر وواضح ومحدد، بالاعتماد التام على نصوص وبنود العقد المرفقة في السياق.

تعليمات الإجابة:
1. أجب عن السؤال مباشرة دون مقدمات طويلة (مثلاً: إذا سأل عن القيمة المالية أو الأجرة، اذكر القيمة وطريقة السداد ومواعيدها بدقة).
2. استخدم لغة عربية سليمة وواضحة ومنظمة في نقاط إذا كانت الإجابة تتضمن شروطاً متعددة.
3. اذكر رقم البند أو المادة ورقم الصفحة التي وردت فيها المعلومة.
4. إذا كان السؤال عن شيء لم يرد ذكره مطلقاً في بنود العقد المتاحة، قل بوضوح: "لم يرد في نصوص العقد المرفوع أي بند ينص على ذلك".
"""


class LegalAugmentor:
    """Handles the Augmentation phase of the RAG pipeline."""

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_blocks = []

        for idx, doc in enumerate(retrieved_docs, start=1):
            text = doc.get("text", doc.get("page_content", ""))
            pages = doc.get("source_pages", [])
            page_str = f"الصفحة: {pages}" if pages else ""
            
            context_blocks.append(
                f"--- [مقطع من العقد {idx}] ({page_str}) ---\n{text}"
            )

        return "\n\n".join(context_blocks)

    def augment(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        context = self.build_context(retrieved_docs)

        user_prompt = f"""نصوص وبنود العقد ذات الصلة:
{context}

سؤال المستخدم:
{query}

أجب عن سؤال المستخدم إجابة مباشرة ودقيقة استناداً إلى نصوص العقد أعلاه:"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]