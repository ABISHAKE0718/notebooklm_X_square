import streamlit as st

import rag_engine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NotebookLM X",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "active_source" not in st.session_state:

    st.session_state.active_source = None


if "answer_mode" not in st.session_state:

    st.session_state.answer_mode = "Notebook"


# ============================================================
# MIGRATE OLD PDF LOCATION
# ============================================================

def sync_old_pdfs():

    old_data_dir = rag_engine.Path(
        "data"
    )

    if not old_data_dir.exists():

        return

    old_pdfs = list(
        old_data_dir.glob("*.pdf")
    )

    for old_pdf in old_pdfs:

        destination = (
            rag_engine.SOURCES_DIR
            / old_pdf.name
        )

        if not destination.exists():

            try:

                destination.write_bytes(
                    old_pdf.read_bytes()
                )

            except Exception:

                pass


sync_old_pdfs()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📚 NotebookLM X"
)

st.caption(
    "Local PDF RAG + Persistent Chat + User Memory"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📚 Sources"
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_files = st.file_uploader(
        "➕ Add Resource",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            save_path = (
                rag_engine.SOURCES_DIR
                / uploaded_file.name
            )

            try:

                save_path.write_bytes(
                    uploaded_file.getvalue()
                )

                with st.spinner(
                    f"Indexing {uploaded_file.name}..."
                ):

                    rag_engine.process_pdf(
                        save_path,
                        rebuild=True
                    )

                st.success(
                    f"Added: {uploaded_file.name}"
                )

                # Uploaded PDF becomes active
                st.session_state.active_source = (
                    uploaded_file.name
                )

            except Exception as error:

                st.error(
                    f"Failed to process "
                    f"{uploaded_file.name}: "
                    f"{error}"
                )

    # ========================================================
    # SOURCES
    # ========================================================

    sources = (
        rag_engine.get_source_names()
    )

    st.divider()

    if sources:

        # ----------------------------------------------------
        # Make sure source is valid
        # ----------------------------------------------------

        if (
            st.session_state.active_source
            not in sources
        ):

            st.session_state.active_source = (
                sources[0]
            )

        # ----------------------------------------------------
        # ACTIVE PDF
        # ----------------------------------------------------

        selected_source = st.selectbox(
            "📄 Active PDF",
            sources,
            index=sources.index(
                st.session_state.active_source
            )
        )

        st.session_state.active_source = (
            selected_source
        )

        st.success(
            f"🟢 Active Source\n\n"
            f"{selected_source}"
        )

        st.caption(
            "Notebook mode searches only this PDF."
        )

        # ----------------------------------------------------
        # DELETE PDF
        # ----------------------------------------------------

        st.subheader(
            "🗑️ Delete Resource"
        )

        delete_choice = st.selectbox(
            "Select PDF",
            ["None"] + sources
        )

        if delete_choice != "None":

            if st.button(
                "🗑️ Delete Selected PDF",
                use_container_width=True
            ):

                try:

                    rag_engine.delete_source(
                        delete_choice
                    )

                    remaining = (
                        rag_engine.get_source_names()
                    )

                    if (
                        st.session_state.active_source
                        == delete_choice
                    ):

                        if remaining:

                            st.session_state.active_source = (
                                remaining[0]
                            )

                        else:

                            st.session_state.active_source = (
                                None
                            )

                    st.success(
                        "PDF deleted."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Delete failed: {error}"
                    )

        # ----------------------------------------------------
        # REBUILD INDEX
        # ----------------------------------------------------

        if st.button(
            "🔄 Rebuild Active Index",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Rebuilding index..."
                ):

                    rag_engine.rebuild_source(
                        st.session_state.active_source
                    )

                st.success(
                    "Index rebuilt successfully."
                )

            except Exception as error:

                st.error(
                    f"Rebuild failed: {error}"
                )

    else:

        st.info(
            "Upload a PDF to start."
        )

    st.divider()

    # ========================================================
    # ANSWER MODE
    # ========================================================

    st.subheader(
        "⚙️ Answer Mode"
    )

    st.session_state.answer_mode = st.radio(
        "Mode",
        [
            "Notebook",
            "Auto",
            "General AI"
        ],
        index=[
            "Notebook",
            "Auto",
            "General AI"
        ].index(
            st.session_state.answer_mode
        )
    )

    if st.session_state.answer_mode == "Notebook":

        st.caption(
            "📄 Uses active PDF only."
        )

    elif st.session_state.answer_mode == "Auto":

        st.caption(
            "📄 PDF first, General AI fallback."
        )

    else:

        st.caption(
            "🤖 General AI only."
        )

    st.divider()

    # ========================================================
    # CLEAR CURRENT CHAT
    # ========================================================

    if st.button(
        "🧹 Clear Current Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # ========================================================
    # SAVED HISTORY
    # ========================================================

    st.subheader(
        "🕘 Previous History"
    )

    records = (
        rag_engine.get_substantive_history()
    )

    if not records:

        st.caption(
            "No saved questions yet."
        )

    else:

        for record in reversed(
            records[-10:]
        ):

            source = (
                record.get("source")
                or
                "General AI"
            )

            with st.expander(
                f"📄 {source}"
            ):

                st.caption(
                    record.get(
                        "timestamp",
                        ""
                    )
                )

                st.markdown(
                    "**Question:**"
                )

                st.write(
                    record.get(
                        "question",
                        ""
                    )
                )

                st.markdown(
                    "**Answer:**"
                )

                st.write(
                    record.get(
                        "answer",
                        ""
                    )
                )


# ============================================================
# MAIN ACTIVE SOURCE
# ============================================================

if st.session_state.active_source:

    st.info(
        f"📄 Active PDF: "
        f"**{st.session_state.active_source}**"
    )

else:

    st.warning(
        "No PDF selected. Upload and select a PDF."
    )


# ============================================================
# CURRENT CHAT
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        ""
    )

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    if role == "user":

        with st.chat_message(
            "user"
        ):

            st.markdown(
                content
            )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    elif role == "assistant":

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                content
            )

            mode = message.get(
                "mode",
                ""
            )

            if mode:

                st.caption(
                    f"Mode: {mode}"
                )

            source_list = message.get(
                "sources",
                []
            )

            if source_list:

                citation_text = []

                for source in source_list:

                    citation_text.append(
                        f"📄 "
                        f"{source['source']} "
                        f"— Page "
                        f"{source['page']}"
                    )

                st.caption(
                    "Sources: "
                    +
                    " | ".join(
                        citation_text
                    )
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question..."
)


if question:

    # ========================================================
    # SHOW USER QUESTION
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # ========================================================
    # UPDATE USER MEMORY
    # ========================================================

    try:

        rag_engine.update_user_memory(
            question
        )

    except Exception as error:

        st.warning(
            f"Memory update warning: {error}"
        )

    # ========================================================
    # CURRENT CHAT HISTORY
    # ========================================================

    current_history = (
        st.session_state.messages[:-1]
    )

    # ========================================================
    # PREPARE QUESTION
    # ========================================================

    try:

        plan = rag_engine.prepare_question(
            question=question,
            active_source=(
                st.session_state.active_source
            ),
            mode=(
                st.session_state.answer_mode
            ),
            current_history=current_history
        )

    except Exception as error:

        answer = (
            "Question processing failed.\n\n"
            f"Error: `{error}`"
        )

        with st.chat_message(
            "assistant"
        ):

            st.error(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [],
                "mode": "Error"
            }
        )

        st.stop()

    # ========================================================
    # EXPLICIT PDF SWITCH
    # ========================================================

    source_used = plan.get(
        "source_used"
    )

    if (
        source_used
        and
        source_used in sources
    ):

        st.session_state.active_source = (
            source_used
        )

    # ========================================================
    # DIRECT / HISTORY / MEMORY
    # ========================================================

    if plan["type"] in [
        "direct",
        "history",
        "memory"
    ]:

        answer = plan["answer"]

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

            st.caption(
                f"Mode: "
                f"{plan.get('mode', '')}"
            )

        # ----------------------------------------------------
        # CURRENT CHAT
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": plan.get(
                    "sources",
                    []
                ),
                "mode": plan.get(
                    "mode",
                    ""
                )
            }
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        if plan.get(
            "save_history",
            True
        ):

            rag_engine.append_history_record(
                question=question,
                answer=answer,
                source=plan.get(
                    "source_used"
                ),
                mode=plan.get(
                    "mode",
                    ""
                ),
                citations=plan.get(
                    "sources",
                    []
                ),
                kind=plan.get(
                    "kind",
                    "normal"
                )
            )

    # ========================================================
    # LLM / RAG
    # ========================================================

    elif plan["type"] == "llm":

        answer = ""

        with st.chat_message(
            "assistant"
        ):

            try:

                answer = st.write_stream(
                    rag_engine.generate_stream(
                        plan["prompt"],
                        plan.get(
                            "num_predict",
                            rag_engine.NORMAL_NUM_PREDICT
                        )
                    )
                )

                if not isinstance(
                    answer,
                    str
                ):

                    answer = str(
                        answer
                    )

            except Exception as error:

                answer = (
                    "Ollama generation failed.\n\n"
                    f"Error: `{error}`"
                )

                st.error(
                    answer
                )

            # ------------------------------------------------
            # MODE
            # ------------------------------------------------

            st.caption(
                f"Mode: "
                f"{plan.get('mode', '')}"
            )

            # ------------------------------------------------
            # PDF CITATIONS
            # ------------------------------------------------

            source_list = plan.get(
                "sources",
                []
            )

            if source_list:

                citation_text = []

                for source in source_list:

                    citation_text.append(
                        f"📄 "
                        f"{source['source']} "
                        f"— Page "
                        f"{source['page']}"
                    )

                st.caption(
                    "Sources: "
                    +
                    " | ".join(
                        citation_text
                    )
                )

        # ====================================================
        # SAVE CURRENT CHAT
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": plan.get(
                    "sources",
                    []
                ),
                "mode": plan.get(
                    "mode",
                    ""
                )
            }
        )

        # ====================================================
        # SAVE PERMANENT HISTORY
        # ====================================================

        if plan.get(
            "save_history",
            True
        ):

            rag_engine.append_history_record(
                question=question,
                answer=answer,
                source=plan.get(
                    "source_used"
                ),
                mode=plan.get(
                    "mode",
                    ""
                ),
                citations=plan.get(
                    "sources",
                    []
                ),
                kind=plan.get(
                    "kind",
                    "normal"
                )
            )