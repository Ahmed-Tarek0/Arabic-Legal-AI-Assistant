import streamlit as st

from retriever import LegalRetriever
from augmentor import LegalAugmentor
from generator import LegalGenerator
from rag_pipeline import LegalRAGPipeline


st.set_page_config(
    page_title="Arabic Legal AI Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Arabic Legal AI Assistant")
st.write("اسأل سؤالك القانوني وسيبحث النظام في المستندات القانونية.")


@st.cache_resource
def load_pipeline():
    retriever = LegalRetriever()
    augmentor = LegalAugmentor()
    generator = LegalGenerator()

    return LegalRAGPipeline(
        retriever=retriever,
        augmentor=augmentor,
        generator=generator
    )


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error("حدث خطأ أثناء تحميل النظام.")
    st.exception(e)
    st.stop()


query = st.text_area(
    "اكتب سؤالك القانوني:",
    placeholder="مثال: ما هي شروط صحة عقد البيع؟",
    height=120
)


if st.button("🔍 اسأل", use_container_width=True):

    if not query.strip():
        st.warning("من فضلك اكتب سؤالًا أولًا.")

    else:
        with st.spinner("جاري البحث وتحليل السؤال..."):

            try:
                answer, documents = pipeline.generate_answer(
                    query,
                    top_k=3
                )

                st.subheader("📌 الإجابة القانونية")
                st.write(answer)

                if documents:
                    st.subheader("📚 المصادر")

                    for i, doc in enumerate(documents, 1):

                        with st.expander(f"المصدر {i}"):

                            st.write(
                                f"**Score:** {doc.get('score', 'N/A')}"
                            )

                            st.write(
                                f"**Document:** {doc.get('doc_id', 'N/A')}"
                            )

                            st.write(
                                f"**Pages:** {doc.get('source_pages', 'N/A')}"
                            )

                            st.write(
                                doc.get('text', '')
                            )

            except Exception as e:
                st.error("حدث خطأ أثناء معالجة السؤال.")
                st.exception(e)