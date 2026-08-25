import os
from retriever import LegalRetriever
# يمكنك استخدام groq أو google.generativeai أو openai
from groq import Groq 

class LegalRAGGenerator:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def augment_prompt(self, query: str, retrieved_chunks: list) -> str:
        context_str = "\n\n---\n\n".join(retrieved_chunks)
        
        system_prompt = f"""أنت مساعد قانوني ذكي ومتخصص في القانون العربي. 
أجب على سؤال المستخدم بناءً على النصوص القانونية المرفقة فقط. 
إذا لم تجد الإجابة في النصوص المرفقة، قل بوضوح: "المعلومة غير متوفرة في المستندات المتاحة".

النصوص القانونية المسترجعة:
{context_str}

سؤال المستخدم:
{query}

الإجابة القانونية الدقيقة:"""
        
        return system_prompt

    def generate_answer(self, query: str, top_k: int = 3):
        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # 2. Augmentation
        augmented_prompt = self.augment_prompt(query, retrieved_docs)
        
        # 3. Generation
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=0.2 # حرارة منخفضة لضمان الدقة القانونية وعدم التأليف
        )
        
        return response.choices[0].message.content, retrieved_docs
