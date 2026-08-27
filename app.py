"""
Simple, Clean & Fast Streamlit Interface for Arabic Contract Q&A Assistant.
"""

from __future__ import annotations

import streamlit as st

from augmentor import LegalAugmentor
from document_processor import process_contract_dynamically
from generator import LegalGenerator
from rag_pipeline import LegalRAGPipeline
from sample_contracts import SAMPLE_CONTRACTS

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="المساعد القانوني الذكي لتحليل العقود",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Clean RTL Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, input, button, select, textarea {
        font-family: 'Cairo', sans-serif !important;
        direction: rtl;
        text-align: right;
    }

    /* Header Banner */
    .app-header {
        text-align: center;
        padding: 1.6rem 1.2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d9488 100%);
        color: white;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .app-header h2 {
        color: #ffffff !important;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        font-size: 1.8rem;
    }
    .app-header p {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Evidence Box */
    .source-box {
        background-color: rgba(13, 148, 136, 0.08);
        border-right: 4px solid #0d9488;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.7;
    }

    /* RTL sidebar overrides */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stChatInput"] {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "contract_name" not in st.session_state:
    st.session_state.contract_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_file_id" not in st.session_state:
    st.session_state.active_file_id = None


# ==============================================================================
# Sidebar UI (Simple & Clean)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚖️ إدارة العقد")
    st.markdown("---")

    st.markdown("#### 📂 1. ارفع ملف العقد")
    uploaded_file = st.file_uploader(
        "اختر ملف العقد (PDF / Word / TXT):",
        type=["pdf", "docx", "txt", "text"],
        help="ارفع أي عقد أو اتفاقية للاستفسار عنها",
    )

    st.markdown("#### أو اختر نموذج عقد جاهز للتجربة:")
    sample_choice = st.selectbox(
        "عقود نموذجية:",
        ["-- اختر نموذجاً للتجربة --"] + list(SAMPLE_CONTRACTS.keys()),
        index=0,
    )

    load_sample = False
    if sample_choice != "-- اختر نموذجاً للتجربة --":
        load_sample = st.button("📥 تحميل هذا العقد", use_container_width=True)

    # Ingestion Logic
    file_to_process = None
    filename_to_process = ""

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.active_file_id != file_id:
            file_to_process = uploaded_file
            filename_to_process = uploaded_file.name
            st.session_state.active_file_id = file_id

    elif load_sample and sample_choice in SAMPLE_CONTRACTS:
        file_to_process = SAMPLE_CONTRACTS[sample_choice].encode("utf-8")
        filename_to_process = f"{sample_choice}.txt"
        st.session_state.active_file_id = filename_to_process

    if file_to_process is not None:
        with st.spinner("⏳ جاري قراءة نصوص العقد وبناء الفهرس الدلالي..."):
            try:
                retriever, chunks, pages = process_contract_dynamically(
                    file_source=file_to_process,
                    filename=filename_to_process,
                )
                generator = LegalGenerator()
                st.session_state.pipeline = LegalRAGPipeline(
                    retriever=retriever,
                    augmentor=LegalAugmentor(),
                    generator=generator,
                )
                st.session_state.contract_name = filename_to_process
                st.session_state.messages = []
                st.success(f"✅ تم تحميل العقد: {filename_to_process}")
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة الملف: {e}")

    # Ensure generator & augmentor are always refreshed
    if st.session_state.pipeline:
        st.session_state.pipeline.generator = LegalGenerator()
        st.session_state.pipeline.augmentor = LegalAugmentor()

    st.markdown("---")
    if st.button("🗑️ تفريغ العقد والبدء من جديد", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.contract_name = None
        st.session_state.messages = []
        st.session_state.active_file_id = None
        st.rerun()


# ==============================================================================
# Main Page (Chat Interface)
# ==============================================================================

# Header
st.markdown(
    """
    <div class="app-header">
        <h2>⚖️ المساعد القانوني الذكي</h2>
        <p>ارفع أي عقد واسأل عن أي بند أو شرط أو التزام، وسيقوم الذكاء الاصطناعي بالإجابة مباشرة بدقة واختصار.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.pipeline:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1.5rem; border: 2px dashed #94a3b8; border-radius: 12px; margin-top: 1rem;">
            <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">📄</div>
            <h3 style="color: #0f172a; margin-bottom: 0.5rem; font-weight: 700;">ابدأ برفع ملف العقد من القائمة الجانبية</h3>
            <p style="color: #64748b;">(يدعم ملفات PDF و Word و TXT أو النماذج الجاهزة للتجربة)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(f"📄 **العقد النشط:** `{st.session_state.contract_name}`")

    # Display Messages History
    for msg in st.session_state.messages:
        avatar = "🧑‍💼" if msg["role"] == "user" else "⚖️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            # Show Reference Evidence Expanders
            if msg.get("docs"):
                with st.expander("🔍 نص البند المرجعي من العقد", expanded=False):
                    for doc in msg["docs"]:
                        pages = doc.get("source_pages", [])
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <b>📌 الصفحة {pages}:</b><br>
                                {doc.get('text', '')}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input Box
    user_query = st.chat_input("اكتب سؤالك عن العقد هنا (مثلاً: ما هي قيمة الإيجار؟ أو ما هي مدة العقد وشروط فسخه؟)...")

    if user_query:
        # Add user query to state & UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_query)

        # Assistant Generation
        with st.chat_message("assistant", avatar="⚖️"):
            response_placeholder = st.empty()
            with st.spinner("🤖 جاري قراءة العقد واستنباط الإجابة المباشرة..."):
                try:
                    stream_iter, retrieved_docs = st.session_state.pipeline.generate_answer_stream(
                        query=user_query,
                        top_k=4,
                    )

                    full_answer = ""
                    for chunk in stream_iter:
                        full_answer += chunk
                        response_placeholder.markdown(full_answer + "▌")
                    
                    response_placeholder.markdown(full_answer)

                    # Show source snippets
                    if retrieved_docs:
                        with st.expander("🔍 نص البند المرجعي من العقد", expanded=False):
                            for doc in retrieved_docs:
                                pages = doc.get("source_pages", [])
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <b>📌 الصفحة {pages}:</b><br>
                                        {doc.get('text', '')}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "docs": retrieved_docs,
                    })

                except Exception as e:
                    err_msg = f"❌ تعذر توليد الإجابة: {e}"
                    response_placeholder.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "docs": [],
                    })
