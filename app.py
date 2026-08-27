"""
Arabic Legal AI Assistant - Fast, Clean & Interactive Contract Analysis.
Supports Drag & Drop file upload directly from device, instant indexing, and real-time streaming Q&A.
"""

from __future__ import annotations

import streamlit as st

from augmentor import LegalAugmentor
from document_processor import process_contract_dynamically
from generator import LegalGenerator
from rag_pipeline import LegalRAGPipeline

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="المساعد القانوني الذكي لتحليل العقود",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Clean RTL & Drag-and-Drop Styling
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

    /* Drag & Drop Upload Zone Styling */
    [data-testid="stFileUploader"] {
        direction: rtl;
        text-align: center;
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed #0d9488 !important;
        background-color: rgba(13, 148, 136, 0.04) !important;
        border-radius: 14px !important;
        padding: 2rem 1rem !important;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #0f766e !important;
        background-color: rgba(13, 148, 136, 0.08) !important;
    }

    /* Active Contract Badge */
    .contract-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.2rem;
        color: #166534;
        font-weight: 600;
        font-size: 0.95rem;
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

    /* RTL overrides */
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
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# ==============================================================================
# Sidebar UI (Clean Options)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚖️ إدارة العقد")
    st.markdown("---")

    sidebar_file = st.file_uploader(
        "رفع عقد جديد:",
        type=["pdf", "docx", "txt", "text"],
        key="sidebar_uploader",
        help="اسحب أو اختر ملف عقد جديد من جهازك",
    )

    if st.session_state.contract_name:
        st.markdown(f"📄 **العقد الحالي:** `{st.session_state.contract_name}`")
        if st.session_state.doc_stats:
            st.caption(f"📑 عدد الصفحات: {st.session_state.doc_stats.get('pages', 1)} | 🧩 المقاطع: {st.session_state.doc_stats.get('chunks', 1)}")

    st.markdown("---")
    if st.button("🗑️ تفريغ العقد والبدء من جديد", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.contract_name = None
        st.session_state.messages = []
        st.session_state.active_file_id = None
        st.session_state.doc_stats = {}
        st.rerun()


# ==============================================================================
# Main Page
# ==============================================================================

# Header
st.markdown(
    """
    <div class="app-header">
        <h2>⚖️ المساعد القانوني الذكي</h2>
        <p>ارفع أي عقد واسأل عن أي بند أو شرط أو التزام، وسيجيبك الذكاء الاصطناعي بدقة فورية وسرعة فائقة.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# File Ingestion handling (Main Dropzone or Sidebar)
uploaded_file = None

if not st.session_state.pipeline:
    st.markdown("### 📂 اسحب ملف العقد وأفلته هنا للبدء:")
    main_file = st.file_uploader(
        "اسحب الملف من جهازك وأفلته هنا مباشرة (PDF / Word / TXT):",
        type=["pdf", "docx", "txt", "text"],
        key="main_uploader",
        label_visibility="collapsed",
    )
    uploaded_file = main_file or sidebar_file
else:
    uploaded_file = sidebar_file

# Process File if new
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.active_file_id != file_id:
        st.session_state.active_file_id = file_id
        with st.spinner("⏳ جاري قراءة نصوص العقد وفهرستها في ثوانٍ..."):
            try:
                retriever, chunks, pages = process_contract_dynamically(
                    file_source=uploaded_file,
                    filename=uploaded_file.name,
                )
                generator = LegalGenerator()
                st.session_state.pipeline = LegalRAGPipeline(
                    retriever=retriever,
                    augmentor=LegalAugmentor(),
                    generator=generator,
                )
                st.session_state.contract_name = uploaded_file.name
                st.session_state.doc_stats = {"pages": len(pages), "chunks": len(chunks)}
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة الملف: {e}")

# ==============================================================================
# Chat Interface (When Contract is Loaded)
# ==============================================================================
if st.session_state.pipeline:
    # Contract Info Header
    st.markdown(
        f"""
        <div class="contract-badge">
            <span>📄 <b>العقد المرفوع:</b> {st.session_state.contract_name}</span>
            <span>⚡ جاهز للإجابة الفورية</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ensure pipeline generator & augmentor are refreshed
    st.session_state.pipeline.generator = LegalGenerator()
    st.session_state.pipeline.augmentor = LegalAugmentor()

    # Display Chat History
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

        # Assistant Generation with smooth streaming
        with st.chat_message("assistant", avatar="⚖️"):
            try:
                # Instant retrieval (< 0.05s) + streaming generation
                stream_iter, retrieved_docs = st.session_state.pipeline.generate_answer_stream(
                    query=user_query,
                    top_k=4,
                )

                full_answer = st.write_stream(stream_iter)

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
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "docs": [],
                })
