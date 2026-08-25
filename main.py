from retriever import LegalRetriever
from generator import LegalRAGGenerator

def main():
    print("🤖 جاري تحميل نظام المساعد القانوني...")
    retriever = LegalRetriever()
    rag_chain = LegalRAGGenerator(retriever)
    
    while True:
        query = input("\nادخل سؤالك القانوني (أو اكتب 'exit' للخروج): ")
        if query.lower() == 'exit':
            break
            
        answer, docs = rag_chain.generate_answer(query)
        print("\n--- الإجابة ---")
        print(answer)

if __name__ == "__main__":
    main()
