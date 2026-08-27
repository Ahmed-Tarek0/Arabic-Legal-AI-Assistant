"""
Augmentation module for Arabic Legal Assistant.
Builds clean, beautifully formatted prompts focused on direct, well-structured legal answers.
"""

from typing import Any, Dict, List

SYSTEM_PROMPT = """أنت مستشار قانوني ذكي وموجز، متخصص في فحص العقود والاتفاقيات بدقة متناهية.

مهمتك:
تقديم إجابة قانونية واضحة، مباشرة، ومنسقة بأعلى درجات الترتيب والجمال بناءً على نصوص العقد المرفقة فقط.

قواعد التنسيق وجودة الإخراج:
1. الإيجاز المباشر: أجب عن السؤال المطلوب مباشرة دون أي مقدمات رتيبة أو حشو كلامي.
2. التنسيق النظيف:
   - استخدم النقاط المباشرة (•) أو الترقيم (1, 2, 3) عند وجود تفاصيل أو شروط متعددة.
   - ضع الكلمات الجوهرية (الأسماء، القيم المالية، المدد الزمنية، الشروط) بخط عريض (**مثل هذا**).
   - اذكر المرجع في سطر أنيق ومستقل أسفل الإجابة أو البند كالتالي:
     📌 **المرجع:** البند [اسم/رقم البند] - صفحة [رقم الصفحة]
   - تجنب تماماً النقاط الفرعية المتداخلة غير المنسقة.
3. الدقة القانونية: إذا لم يرد ذكر السؤال في بنود العقد المرفقة، قل بوضوح: "لم يرد في بنود العقد المرفوع ما ينص على ذلك".
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
                f"--- [بند من العقد {idx}] ({page_str}) ---\n{text}"
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

أجب عن سؤال المستخدم إجابة مباشرة ودقيقة ومنسقة استناداً إلى نصوص العقد أعلاه:"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]