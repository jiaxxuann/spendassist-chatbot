"""
retrieval/vector_store.py — RAG pipeline for the Company Z expense policy.

Responsibilities:
- Read data/expense_policy.md and split it into per-clause LangChain Documents
  using regex-based structural chunking (not a generic character splitter).
- Store the exact clause identifier (e.g. "Clause 2.1") in each Document's
  metadata so the LangGraph agent can surface precise citations.
- Build and persist a local Chroma vector database at data/chroma_db using
  the sentence-transformers/all-MiniLM-L6-v2 embedding model.
- Expose get_retriever() for the agent to query at runtime.
"""

import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent          # project root
_POLICY_PATH = _ROOT / "data" / "expense_policy.md"
_CHROMA_DIR = str(_ROOT / "data" / "chroma_db")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "expense_policy"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clause pattern
#
# Matches lines of the form:
#   - **Clause 1.2 (Accommodation):** Hotel stays are capped …
#
# Capture groups:
#   1 → clause_id   e.g. "1.2"
#   2 → title       e.g. "Accommodation"
#   3 → body        e.g. "Hotel stays are capped …"
# ---------------------------------------------------------------------------

_CLAUSE_RE = re.compile(
    r"-\s+\*\*Clause\s+([\d.]+)\s+\(([^)]+)\):\*\*\s+(.*)"
)

# Section heading pattern used to track the current section for each clause.
_SECTION_RE = re.compile(r"^##\s+(.+)")


def _parse_policy(policy_path: Path) -> list[Document]:
    """Parse expense_policy.md into one LangChain Document per clause.

    Uses regex-based structural chunking: each ``- **Clause X.Y`` line
    becomes its own chunk.  The clause identifier, title, and parent section
    are stored as Document metadata so they can be surfaced as citations.

    Args:
        policy_path: Absolute path to the Markdown policy file.

    Returns:
        A list of :class:`langchain_core.documents.Document` objects, one per
        clause, with the following metadata keys:

        - ``clause_id``  – e.g. ``"Clause 2.1"``
        - ``title``      – e.g. ``"Daily Per Diem"``
        - ``section``    – e.g. ``"Section 2: Meal Allowances"``
        - ``source``     – relative path to the source file
    """
    raw_text = policy_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    documents: list[Document] = []
    current_section = "Unknown Section"

    for line in lines:
        # Track the current section heading
        section_match = _SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        # Attempt to match a clause line
        clause_match = _CLAUSE_RE.match(line)
        if not clause_match:
            continue

        clause_num = clause_match.group(1).strip()   # e.g. "2.1"
        title      = clause_match.group(2).strip()   # e.g. "Daily Per Diem"
        body       = clause_match.group(3).strip()   # rule text

        clause_id = f"Clause {clause_num}"           # canonical citation form

        # The page_content includes the identifier so the LLM can quote it
        # verbatim in its answer without needing extra post-processing.
        page_content = f"{clause_id} ({title}): {body}"

        metadata = {
            "clause_id": clause_id,
            "title":     title,
            "section":   current_section,
            "source":    str(policy_path.relative_to(_ROOT)),
        }

        documents.append(Document(page_content=page_content, metadata=metadata))
        logger.debug("Parsed %s — '%s'", clause_id, title)

    logger.info("Parsed %d clause chunks from %s", len(documents), policy_path.name)
    return documents


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the local HuggingFace embedding model.

    Downloads ``sentence-transformers/all-MiniLM-L6-v2`` on first run and
    caches it locally via the HuggingFace hub cache.  All subsequent calls
    use the cached weights — no network required.
    """
    logger.info("Loading embedding model '%s' …", _EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_vector_store() -> Chroma:
    """Build the Chroma vector store from the policy document and persist it.

    This is a one-time (or refresh) operation.  Run it explicitly to create
    or rebuild ``data/chroma_db``.  Existing collections with the same name
    are replaced so the store always reflects the current policy document.

    Returns:
        The initialised :class:`langchain_chroma.Chroma` instance.

    Raises:
        FileNotFoundError: If ``data/expense_policy.md`` is not found.
        RuntimeError: If the Chroma persist directory cannot be created.
    """
    if not _POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Policy document not found at '{_POLICY_PATH}'. "
            "Ensure data/expense_policy.md exists before initialising the vector store."
        )

    docs = _parse_policy(_POLICY_PATH)
    embeddings = _build_embeddings()

    logger.info(
        "Building Chroma collection '%s' at '%s' …",
        _COLLECTION_NAME,
        _CHROMA_DIR,
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=_COLLECTION_NAME,
        persist_directory=_CHROMA_DIR,
    )

    logger.info(
        "Vector store initialised with %d documents. Persisted to '%s'.",
        len(docs),
        _CHROMA_DIR,
    )
    return vector_store


def get_retriever(
    k: int = 3,
    score_threshold: float = 0.0,
) -> VectorStoreRetriever:
    """Load the persisted Chroma store and return a retriever for the agent.

    Connects to the existing ``data/chroma_db`` directory without rebuilding
    the index.  Call :func:`initialize_vector_store` first if the database
    does not yet exist.

    Args:
        k: Number of top-matching clauses to return per query (default: 3).
        score_threshold: Minimum relevance score in ``[0, 1]`` for returned
            documents when using ``similarity_score_threshold`` search type.
            Has no effect when using the default ``similarity`` search type.

    Returns:
        A :class:`langchain_core.vectorstores.VectorStoreRetriever` configured
        for similarity search.  The retrieved Documents carry ``clause_id``
        metadata that the agent must append verbatim to its final answer as
        a citation.

    Raises:
        RuntimeError: If ``data/chroma_db`` does not exist (store not yet
            initialised).
    """
    chroma_path = Path(_CHROMA_DIR)
    if not chroma_path.exists():
        raise RuntimeError(
            f"Chroma DB not found at '{_CHROMA_DIR}'. "
            "Run initialize_vector_store() first to build the index."
        )

    embeddings = _build_embeddings()

    vector_store = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=_CHROMA_DIR,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    logger.info(
        "Retriever ready — collection '%s', top_k=%d.",
        _COLLECTION_NAME,
        k,
    )
    return retriever


# ---------------------------------------------------------------------------
# CLI helper — run directly to (re)build the index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Initialising SpendAssist vector store …")
    store = initialize_vector_store()

    # Smoke-test: retrieve the top clause for a sample query
    test_query = "What is the hotel accommodation limit per night?"
    retriever = store.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke(test_query)

    print(f"\nSmoke-test query: '{test_query}'")
    print(f"Top {len(results)} result(s):\n")
    for doc in results:
        print(f"  [{doc.metadata['clause_id']}] {doc.page_content}")
        print(f"  Metadata: {doc.metadata}\n")

