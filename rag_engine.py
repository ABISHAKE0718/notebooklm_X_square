import re
import json
import hashlib
import shutil
import difflib
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import ollama

from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

OLLAMA_MODEL = "qwen2.5:0.5b-instruct"

SOURCES_DIR = Path("data/sources")

INDEX_ROOT = Path("source_indexes")

HISTORY_FILE = Path("chat_history.json")

MEMORY_FILE = Path("memory.json")

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

RETRIEVAL_K = 8

FINAL_TOP_K = 3

MAX_CHUNK_CHARS = 1800

NUM_CTX = 4096

NORMAL_NUM_PREDICT = 220


# ============================================================
# CREATE DIRECTORIES
# ============================================================

SOURCES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SOURCE FUNCTIONS
# ============================================================

def get_source_files():
    """
    Return all PDFs inside data/sources.
    """

    return sorted(
        SOURCES_DIR.glob("*.pdf"),
        key=lambda p: p.name.lower()
    )


def get_source_names():
    """
    Return PDF filenames only.
    """

    return [
        p.name
        for p in get_source_files()
    ]


def get_source_path(source_name):
    """
    Return complete path for a PDF.
    """

    return SOURCES_DIR / source_name


# ============================================================
# SOURCE ID
# ============================================================

def source_id(source_name):
    """
    Create a stable ID for each PDF.

    Every PDF gets a separate FAISS folder.
    """

    return hashlib.sha1(
        source_name.encode("utf-8")
    ).hexdigest()[:12]


def get_index_path(source_name):
    """
    FAISS directory for one PDF.
    """

    return INDEX_ROOT / source_id(
        source_name
    )


def index_exists(source_name):
    """
    Check whether a PDF has its own FAISS index.
    """

    index_path = get_index_path(
        source_name
    )

    return (
        index_path.exists()
        and
        (index_path / "index.faiss").exists()
        and
        (index_path / "index.pkl").exists()
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_embeddings():

    print(
        "Loading embedding model..."
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# VECTOR STORE
# ============================================================

@lru_cache(maxsize=32)
def load_source_vectorstore(
    source_name
):
    """
    Load the FAISS index belonging ONLY
    to the selected PDF.
    """

    if not index_exists(
        source_name
    ):

        raise FileNotFoundError(
            f"FAISS index does not exist for "
            f"{source_name}"
        )

    index_path = get_index_path(
        source_name
    )

    vectorstore = FAISS.load_local(
        str(index_path),
        load_embeddings(),
        allow_dangerous_deserialization=True
    )

    return vectorstore


def clear_source_cache():

    load_source_vectorstore.cache_clear()


# ============================================================
# RERANKER
# ============================================================

@lru_cache(maxsize=1)
def load_reranker():

    print(
        "Loading reranker model..."
    )

    return CrossEncoder(
        RERANKER_MODEL
    )


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(
    pdf_path,
    rebuild=True
):
    """
    PDF
      ↓
    Extract text
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    FAISS
    """

    pdf_path = Path(
        pdf_path
    )

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    source_name = pdf_path.name

    print(
        f"\nProcessing PDF: {source_name}"
    )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    loader = PyMuPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    if not documents:

        raise ValueError(
            f"No text extracted from "
            f"{source_name}"
        )

    # --------------------------------------------------------
    # Add source metadata
    # --------------------------------------------------------

    for document in documents:

        document.metadata[
            "source_name"
        ] = source_name

        document.metadata[
            "source"
        ] = source_name

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:

        raise ValueError(
            f"No chunks created from "
            f"{source_name}"
        )

    print(
        f"Created {len(chunks)} chunks"
    )

    # --------------------------------------------------------
    # Create separate index folder
    # --------------------------------------------------------

    index_path = get_index_path(
        source_name
    )

    if (
        rebuild
        and
        index_path.exists()
    ):

        shutil.rmtree(
            index_path
        )

    index_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create FAISS
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        load_embeddings()
    )

    vectorstore.save_local(
        str(index_path)
    )

    clear_source_cache()

    print(
        f"FAISS created for: "
        f"{source_name}"
    )

    return {
        "source": source_name,
        "chunks": len(chunks),
        "index_path": str(index_path)
    }


# ============================================================
# REBUILD ONE PDF
# ============================================================

def rebuild_source(
    source_name
):

    pdf_path = get_source_path(
        source_name
    )

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"Source not found: "
            f"{source_name}"
        )

    return process_pdf(
        pdf_path,
        rebuild=True
    )


# ============================================================
# REBUILD ALL PDFS
# ============================================================

def rebuild_all_sources():

    results = []

    for pdf_path in get_source_files():

        try:

            result = process_pdf(
                pdf_path,
                rebuild=True
            )

            results.append(
                result
            )

        except Exception as error:

            results.append(
                {
                    "source": pdf_path.name,
                    "error": str(error)
                }
            )

    return results


# ============================================================
# DELETE PDF
# ============================================================

def delete_source(
    source_name
):

    pdf_path = get_source_path(
        source_name
    )

    index_path = get_index_path(
        source_name
    )

    if pdf_path.exists():

        pdf_path.unlink()

    if index_path.exists():

        shutil.rmtree(
            index_path
        )

    clear_source_cache()

    return True


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text
):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FIND PDF REFERRED TO BY USER
# ============================================================

def find_requested_source(
    question,
    available_sources
):
    """
    Detect an explicitly mentioned PDF.

    Supports:

        politics.pdf
        politics
        politcs.pdf
    """

    if not question:
        return None

    if not available_sources:
        return None

    normalized_question = normalize_text(
        question
    )

    # --------------------------------------------------------
    # Exact filename
    # --------------------------------------------------------

    for source in available_sources:

        source_name = Path(
            source
        ).name

        normalized_name = normalize_text(
            source_name
        )

        if normalized_name in normalized_question:

            return source_name

    # --------------------------------------------------------
    # Exact filename without extension
    # --------------------------------------------------------

    source_stems = {}

    for source in available_sources:

        source_name = Path(
            source
        ).name

        stem = Path(
            source_name
        ).stem

        normalized_stem = normalize_text(
            stem
        )

        source_stems[
            source_name
        ] = normalized_stem

        if (
            normalized_stem
            and
            normalized_stem in normalized_question
        ):

            return source_name

    # --------------------------------------------------------
    # Fuzzy matching
    # --------------------------------------------------------

    words = normalized_question.split()

    possible_words = [
        word
        for word in words
        if len(word) >= 4
    ]

    stems = list(
        source_stems.values()
    )

    for word in possible_words:

        matches = difflib.get_close_matches(
            word,
            stems,
            n=1,
            cutoff=0.80
        )

        if matches:

            matched_stem = matches[0]

            for (
                source_name,
                stem
            ) in source_stems.items():

                if stem == matched_stem:

                    return source_name

    return None


# ============================================================
# ============================================================
# USER MEMORY
# ============================================================
# ============================================================

def load_user_memory():
    """
    Load persistent user memory.
    """

    if not MEMORY_FILE.exists():

        return {}

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

            if isinstance(
                data,
                dict
            ):

                return data

    except Exception:

        pass

    return {}


def save_user_memory(
    memory
):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=2,
            ensure_ascii=False
        )


def extract_name_from_message(
    message
):
    """
    Extract a name when user explicitly states it.

    Examples:

        My name is Abishake
        My name is Abishake M
        Call me Abishake
    """

    patterns = [

        r"\bmy name is\s+(.+?)(?:[.!?,;:]|$)",

        r"\bmy name:\s*(.+?)(?:[.!?,;:]|$)",

        r"\bcall me\s+(.+?)(?:[.!?,;:]|$)",

        r"\byou can call me\s+(.+?)(?:[.!?,;:]|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            name = match.group(
                1
            ).strip()

            # Remove common trailing text
            name = re.sub(
                r"\s+(and|but|because|from|i am|i'm)\b.*$",
                "",
                name,
                flags=re.IGNORECASE
            ).strip()

            name = name.rstrip(
                ".,!?;:"
            )

            if (
                name
                and
                len(name) <= 60
            ):

                return name

    return None


def update_user_memory(
    message
):
    """
    Extract facts explicitly given by the user
    and save them permanently.
    """

    memory = load_user_memory()

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name = extract_name_from_message(
        message
    )

    if name:

        memory["name"] = name

    save_user_memory(
        memory
    )

    return memory


def answer_memory_question(
    question
):
    """
    Answer personal-memory questions directly.
    """

    q = normalize_text(
        question
    )

    memory = load_user_memory()

    # ========================================================
    # NAME
    # ========================================================

    is_name_question = (

        "what is my name" in q

        or
        "whats my name" in q

        or
        "what was my name" in q

        or
        "do you remember my name" in q

        or
        "remember my name" in q

        or
        "what did i tell you my name was" in q

        or
        "what name did i give you" in q
    )

    if is_name_question:

        name = memory.get(
            "name"
        )

        if name:

            return (
                f"Your name is **{name}**."
            )

        # ----------------------------------------------------
        # Fallback to old history
        # ----------------------------------------------------

        records = load_history_records()

        for record in reversed(
            records
        ):

            previous_question = record.get(
                "question",
                ""
            )

            found_name = (
                extract_name_from_message(
                    previous_question
                )
            )

            if found_name:

                memory["name"] = (
                    found_name
                )

                save_user_memory(
                    memory
                )

                return (
                    f"Your name is "
                    f"**{found_name}**."
                )

        return (
            "I don't have your name saved yet. "
            "Tell me: **My name is Abishake.**"
        )

    return None


# ============================================================
# ============================================================
# CHAT HISTORY
# ============================================================
# ============================================================

def load_history_records():
    """
    Load persistent Q&A history.

    Supports both the new format and your
    older role/content format.
    """

    if not HISTORY_FILE.exists():

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

    except Exception:

        return []

    if not isinstance(
        data,
        list
    ):

        return []

    # --------------------------------------------------------
    # New format
    # --------------------------------------------------------

    if all(
        isinstance(item, dict)
        and "question" in item
        and "answer" in item
        for item in data
    ):

        return data

    # --------------------------------------------------------
    # Convert old format
    # --------------------------------------------------------

    converted = []

    pending_question = None

    for item in data:

        if not isinstance(
            item,
            dict
        ):

            continue

        role = item.get(
            "role",
            ""
        )

        content = item.get(
            "content",
            ""
        )

        if role == "user":

            pending_question = (
                content
            )

        elif (
            role == "assistant"
            and
            pending_question is not None
        ):

            converted.append(
                {
                    "timestamp": "",
                    "source": "",
                    "question": pending_question,
                    "answer": content,
                    "mode": "Unknown",
                    "kind": "normal",
                    "citations": []
                }
            )

            pending_question = None

    return converted


def save_history_records(
    records
):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )


def append_history_record(
    question,
    answer,
    source=None,
    mode="Notebook",
    citations=None,
    kind="normal"
):
    """
    Append one Q&A to persistent history.
    """

    records = load_history_records()

    record = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": source or "",
        "question": question,
        "answer": answer,
        "mode": mode,
        "kind": kind,
        "citations": citations or []
    }

    records.append(
        record
    )

    save_history_records(
        records
    )

    return record


def get_substantive_history():
    """
    Return real user questions.

    History commands such as:

        What was my previous question?

    are excluded so they don't become
    the new previous question.
    """

    records = load_history_records()

    return [
        record
        for record in records
        if record.get(
            "kind",
            "normal"
        ) != "history"
    ]


# ============================================================
# HISTORY DETECTION
# ============================================================

def detect_history_intent(
    question
):
    """
    Detect questions about previous conversation.

    These questions are answered directly from
    chat_history.json.
    """

    q = normalize_text(
        question
    )

    # ========================================================
    # QUESTION BEFORE LAST
    # ========================================================

    if (
        "what did i ask before that" in q
        or
        "what was the question before that" in q
        or
        "what was my question before that" in q
        or
        "question before that" in q
    ):

        return "question_before_that"

    # ========================================================
    # ANSWER BEFORE LAST
    # ========================================================

    if (
        "what was the answer before that" in q
        or
        "what was the response before that" in q
        or
        "answer to the question before that" in q
        or
        "response to the question before that" in q
    ):

        return "answer_before_that"

    # ========================================================
    # LAST ANSWER
    # ========================================================

    if (
        "what was the answer to my last question" in q
        or
        "what was the answer to my previous question" in q
        or
        "what was the response to my last question" in q
        or
        "what was the response to my previous question" in q
        or
        "answer to my last question" in q
        or
        "answer to my previous question" in q
        or
        "response to my last question" in q
        or
        "response to my previous question" in q
    ):

        return "last_answer"

    # ========================================================
    # LAST / PREVIOUS QUESTION
    # ========================================================

    if (
        "what was my last question" in q
        or
        "what was my previous question" in q
        or
        "what is my last question" in q
        or
        "what is my previous question" in q
        or
        "what did i ask last" in q
        or
        "what did i ask previously" in q
        or
        "what was the last question" in q
        or
        "what was the previous question" in q
        or
        "what is the previous question" in q
    ):

        return "last_question"

    # ========================================================
    # RECENT QUESTIONS
    # ========================================================

    if (
        "show my previous questions" in q
        or
        "show my recent questions" in q
        or
        "show previous questions" in q
        or
        "show recent questions" in q
        or
        "what questions did i ask" in q
        or
        "my previous questions" in q
        or
        "my recent questions" in q
    ):

        return "recent_questions"

    # ========================================================
    # RECENT CHAT
    # ========================================================

    if (
        "show my chat history" in q
        or
        "show chat history" in q
        or
        "show my conversation history" in q
        or
        "what did we discuss previously" in q
        or
        "what did we discuss before" in q
    ):

        return "recent_history"

    return None


# ============================================================
# ANSWER HISTORY QUESTION
# ============================================================

def answer_history_question(
    question
):
    """
    Answer previous-chat questions using the database file.
    """

    intent = detect_history_intent(
        question
    )

    if intent is None:

        return None

    records = get_substantive_history()

    if not records:

        return (
            "There is no saved conversation "
            "history yet."
        )

    # ========================================================
    # LAST QUESTION
    # ========================================================

    if intent == "last_question":

        record = records[-1]

        source = (
            record.get("source")
            or
            "General AI"
        )

        return (
            "Your previous question was:\n\n"
            f"**{record['question']}**\n\n"
            f"Source: {source}"
        )

    # ========================================================
    # LAST ANSWER
    # ========================================================

    if intent == "last_answer":

        record = records[-1]

        return (
            "The answer to your previous "
            "question was:\n\n"
            f"{record['answer']}"
        )

    # ========================================================
    # QUESTION BEFORE LAST
    # ========================================================

    if intent == "question_before_that":

        if len(records) < 2:

            return (
                "There is only one saved question, "
                "so there is no earlier question."
            )

        record = records[-2]

        return (
            "The question before your previous "
            "question was:\n\n"
            f"**{record['question']}**"
        )

    # ========================================================
    # ANSWER BEFORE LAST
    # ========================================================

    if intent == "answer_before_that":

        if len(records) < 2:

            return (
                "There is only one saved answer."
            )

        record = records[-2]

        return (
            "The answer to the question before "
            "your previous question was:\n\n"
            f"{record['answer']}"
        )

    # ========================================================
    # RECENT QUESTIONS
    # ========================================================

    if intent == "recent_questions":

        recent = records[-10:]

        lines = []

        for number, record in enumerate(
            reversed(recent),
            start=1
        ):

            lines.append(
                f"{number}. "
                f"{record['question']}"
            )

        return (
            "Your recent questions:\n\n"
            + "\n".join(lines)
        )

    # ========================================================
    # RECENT HISTORY
    # ========================================================

    if intent == "recent_history":

        recent = records[-5:]

        output = (
            "Your recent conversation:\n\n"
        )

        for record in reversed(
            recent
        ):

            output += (
                f"**Q:** "
                f"{record['question']}\n\n"
            )

            output += (
                f"**A:** "
                f"{record['answer']}\n\n"
            )

            output += (
                "---\n\n"
            )

        return output

    return None


# ============================================================
# SOURCE-SPECIFIC HISTORY
# ============================================================

def get_recent_source_history(
    source_name,
    limit=3
):

    records = load_history_records()

    matching = []

    for record in records:

        if (
            record.get("source")
            == source_name
        ):

            matching.append(
                record
            )

    return matching[-limit:]


# ============================================================
# RETRIEVAL QUERY
# ============================================================

def build_retrieval_query(
    question,
    current_history=None,
    source_name=None
):
    """
    Use previous conversation when the user
    asks a follow-up question.
    """

    current_history = (
        current_history or []
    )

    q = question.lower()

    followup_words = [
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
        "them",
        "above",
        "previous",
        "earlier",
        "same"
    ]

    is_followup = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            q
        )
        for word in followup_words
    )

    if not is_followup:

        return question

    parts = []

    # --------------------------------------------------------
    # Current conversation
    # --------------------------------------------------------

    for item in current_history[-6:]:

        role = item.get(
            "role",
            ""
        )

        content = item.get(
            "content",
            ""
        )

        if content:

            parts.append(
                f"{role}: {content}"
            )

    # --------------------------------------------------------
    # Persistent history for this PDF
    # --------------------------------------------------------

    if source_name:

        previous_records = (
            get_recent_source_history(
                source_name,
                limit=3
            )
        )

        for record in previous_records:

            parts.append(
                "user: "
                + record["question"]
            )

            parts.append(
                "assistant: "
                + record["answer"]
            )

    history_text = "\n".join(
        parts[-8:]
    )

    if not history_text:

        return question

    return (
        "Previous conversation:\n"
        f"{history_text}\n\n"
        "Current question:\n"
        f"{question}"
    )


# ============================================================
# RETRIEVE FROM ACTIVE PDF
# ============================================================

def retrieve_from_source(
    question,
    source_name,
    k=RETRIEVAL_K
):
    """
    Search ONLY the active PDF.
    """

    if not index_exists(
        source_name
    ):

        process_pdf(
            get_source_path(
                source_name
            ),
            rebuild=True
        )

    vectorstore = (
        load_source_vectorstore(
            source_name
        )
    )

    docs = vectorstore.similarity_search(
        question,
        k=k
    )

    return docs


# ============================================================
# RETRIEVE + RERANK
# ============================================================

def retrieve_and_rerank(
    question,
    source_name,
    current_history=None,
    top_k=FINAL_TOP_K
):
    """
    Retrieve from only the active PDF
    and rerank the chunks.
    """

    retrieval_query = (
        build_retrieval_query(
            question,
            current_history,
            source_name
        )
    )

    docs = retrieve_from_source(
        retrieval_query,
        source_name,
        k=RETRIEVAL_K
    )

    if not docs:

        return []

    reranker = (
        load_reranker()
    )

    pairs = [
        [
            question,
            doc.page_content[
                :MAX_CHUNK_CHARS
            ]
        ]
        for doc in docs
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = list(
        zip(
            scores,
            docs
        )
    )

    ranked.sort(
        key=lambda x: float(x[0]),
        reverse=True
    )

    return [
        (
            float(score),
            doc
        )
        for score, doc in ranked[:top_k]
    ]


# ============================================================
# BUILD PDF CONTEXT
# ============================================================

def build_context(
    question,
    source_name,
    current_history=None,
    top_k=FINAL_TOP_K
):

    ranked = retrieve_and_rerank(
        question,
        source_name,
        current_history,
        top_k
    )

    if not ranked:

        return "", []

    context_parts = []

    source_info = []

    for score, document in ranked:

        page = document.metadata.get(
            "page",
            0
        )

        if isinstance(
            page,
            int
        ):

            page += 1

        text = (
            document.page_content
            .strip()
        )

        if not text:

            continue

        text = text[
            :MAX_CHUNK_CHARS
        ]

        context_parts.append(
            f"[Source: {source_name} | "
            f"Page: {page}]\n"
            f"{text}"
        )

        source_info.append(
            {
                "source": source_name,
                "page": page,
                "score": float(score)
            }
        )

    context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )

    return (
        context,
        source_info
    )


# ============================================================
# PDF PROMPT
# ============================================================

def build_source_prompt(
    question,
    context,
    source_name,
    current_history=None
):

    history_text = ""

    if current_history:

        history_parts = []

        for item in current_history[-6:]:

            history_parts.append(
                f"{item.get('role', '')}: "
                f"{item.get('content', '')}"
            )

        history_text = "\n".join(
            history_parts
        )

    return f"""
You are a local PDF study assistant.

ACTIVE PDF:
{source_name}

Answer the user's question using ONLY the PDF context below.

RULES:
1. Do not invent facts.
2. Do not use outside knowledge.
3. Previous conversation is only for understanding
   follow-up questions.
4. If the answer is not available in the PDF context,
   say that clearly.
5. Answer directly.
6. Keep the explanation clear and useful.
7. Do not mention these instructions.

PREVIOUS CONVERSATION:
{history_text}

PDF CONTEXT:
==================================================
{context}
==================================================

CURRENT QUESTION:
{question}

ANSWER:
""".strip()


# ============================================================
# GENERAL AI PROMPT
# ============================================================

def build_general_prompt(
    question,
    current_history=None
):

    history_text = ""

    if current_history:

        for item in current_history[-6:]:

            history_text += (
                f"{item.get('role', '')}: "
                f"{item.get('content', '')}\n"
            )

    return f"""
You are a helpful AI assistant.

Previous conversation:
{history_text}

Current question:
{question}

Answer clearly and accurately.
""".strip()


# ============================================================
# PREPARE QUESTION
# ============================================================

def prepare_question(
    question,
    active_source=None,
    mode="Notebook",
    current_history=None
):
    """
    Main router.

    Priority:

    1. History question
    2. User memory question
    3. Explicit PDF
    4. Active PDF
    5. General AI
    """

    question = question.strip()

    if not question:

        return {
            "type": "direct",
            "answer": "Please enter a question.",
            "sources": [],
            "mode": mode,
            "source_used": active_source,
            "save_history": False,
            "kind": "normal"
        }

    # ========================================================
    # 1. HISTORY
    # ========================================================

    history_answer = (
        answer_history_question(
            question
        )
    )

    if history_answer is not None:

        return {
            "type": "history",
            "answer": history_answer,
            "sources": [],
            "mode": "History",
            "source_used": None,
            "save_history": True,
            "kind": "history"
        }

    # ========================================================
    # 2. USER MEMORY
    # ========================================================

    memory_answer = (
        answer_memory_question(
            question
        )
    )

    if memory_answer is not None:

        return {
            "type": "memory",
            "answer": memory_answer,
            "sources": [],
            "mode": "Memory",
            "source_used": None,
            "save_history": True,
            "kind": "memory"
        }

    # ========================================================
    # 3. GENERAL AI MODE
    # ========================================================

    if mode == "General AI":

        return {
            "type": "llm",
            "prompt": build_general_prompt(
                question,
                current_history
            ),
            "sources": [],
            "mode": "General AI",
            "source_used": None,
            "num_predict": NORMAL_NUM_PREDICT,
            "save_history": True,
            "kind": "normal"
        }

    # ========================================================
    # 4. DETECT EXPLICIT PDF
    # ========================================================

    available_sources = (
        get_source_names()
    )

    requested_source = (
        find_requested_source(
            question,
            available_sources
        )
    )

    if requested_source:

        selected_source = (
            requested_source
        )

    else:

        selected_source = (
            active_source
        )

    # ========================================================
    # 5. NO PDF
    # ========================================================

    if not selected_source:

        if mode == "Auto":

            return {
                "type": "llm",
                "prompt": build_general_prompt(
                    question,
                    current_history
                ),
                "sources": [],
                "mode": "General AI",
                "source_used": None,
                "num_predict": NORMAL_NUM_PREDICT,
                "save_history": True,
                "kind": "normal"
            }

        return {
            "type": "direct",
            "answer": (
                "Please upload and select a PDF "
                "before asking a document question."
            ),
            "sources": [],
            "mode": "Notebook",
            "source_used": None,
            "save_history": True,
            "kind": "normal"
        }

    # ========================================================
    # 6. BUILD INDEX IF NEEDED
    # ========================================================

    if not index_exists(
        selected_source
    ):

        process_pdf(
            get_source_path(
                selected_source
            ),
            rebuild=True
        )

    # ========================================================
    # 7. RETRIEVE PDF
    # ========================================================

    context, sources = build_context(
        question,
        selected_source,
        current_history,
        FINAL_TOP_K
    )

    # ========================================================
    # 8. NO RELEVANT PDF CONTEXT
    # ========================================================

    if not context.strip():

        if mode == "Auto":

            return {
                "type": "llm",
                "prompt": build_general_prompt(
                    question,
                    current_history
                ),
                "sources": [],
                "mode": "General AI",
                "source_used": selected_source,
                "num_predict": NORMAL_NUM_PREDICT,
                "save_history": True,
                "kind": "normal"
            }

        return {
            "type": "direct",
            "answer": (
                f"I could not find relevant information "
                f"in **{selected_source}**."
            ),
            "sources": [],
            "mode": "Notebook",
            "source_used": selected_source,
            "save_history": True,
            "kind": "normal"
        }

    # ========================================================
    # 9. PDF RAG
    # ========================================================

    return {
        "type": "llm",
        "prompt": build_source_prompt(
            question,
            context,
            selected_source,
            current_history
        ),
        "sources": sources,
        "mode": "Notebook",
        "source_used": selected_source,
        "num_predict": NORMAL_NUM_PREDICT,
        "save_history": True,
        "kind": "normal"
    }


# ============================================================
# OLLAMA GENERATION
# ============================================================

def generate_answer(
    prompt,
    num_predict=NORMAL_NUM_PREDICT
):

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False,
        keep_alive=-1,
        options={
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX
        }
    )

    return response[
        "message"
    ]["content"]


# ============================================================
# OLLAMA STREAMING
# ============================================================

def generate_stream(
    prompt,
    num_predict=NORMAL_NUM_PREDICT
):

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True,
        keep_alive=-1,
        options={
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX
        }
    )

    for chunk in response:

        message = chunk.get(
            "message",
            {}
        )

        content = message.get(
            "content",
            ""
        )

        if content:

            yield content


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(
    question,
    source_name
):

    print("\n====================================")
    print("RETRIEVAL DEBUG")
    print("====================================")

    print(
        f"Question: {question}"
    )

    print(
        f"Source: {source_name}"
    )

    print("------------------------------------")

    ranked = retrieve_and_rerank(
        question,
        source_name,
        None,
        FINAL_TOP_K
    )

    if not ranked:

        print(
            "No results found."
        )

        return

    for number, (
        score,
        document
    ) in enumerate(
        ranked,
        start=1
    ):

        page = document.metadata.get(
            "page",
            0
        )

        if isinstance(
            page,
            int
        ):

            page += 1

        print(
            f"\nResult {number}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            document.page_content[:1000]
        )

        print(
            "------------------------------------"
        )