"""
PART 1 — SETUP

Core memory and agent logic shared by all scripts.

AgentMemory  : persistent ChromaDB-backed vector store using a local
               sentence-transformers model — no API keys required.
MockAgent    : simulates an LLM that grounds its answers in retrieved
               memories, making the poisoning effect clearly visible.
"""

import datetime
import sys
import time
import uuid

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

# ── Constants ──────────────────────────────────────────────────────────────────

PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME = "agent_memory"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB ONNX, cached after first run


class FastEmbedFn(EmbeddingFunction[Documents]):
    """
    ChromaDB-compatible embedding function backed by fastembed + ONNX runtime.
    Uses all-MiniLM-L6-v2 locally — no API keys, no torch MPS.
    """
    def __init__(self, model: str = EMBED_MODEL):
        self._model = TextEmbedding(model)

    def __call__(self, input: Documents) -> Embeddings:
        return [vec.tolist() for vec in self._model.embed(input)]

def _typewrite(text: str, delay: float = 0.03, newline: bool = True) -> None:
    """
    Print text one character at a time to simulate live terminal output.
    ANSI escape sequences (e.g. \033[91m) are written atomically so color
    codes never appear mid-sequence and corrupt the terminal output.
    """
    i = 0
    while i < len(text):
        if text[i] == "\033" and i + 1 < len(text) and text[i + 1] == "[":
            # Consume the full escape sequence: ESC [ ... <letter>
            j = i + 2
            while j < len(text) and not text[j].isalpha():
                j += 1
            j += 1  # include the terminating letter
            sys.stdout.write(text[i:j])
            sys.stdout.flush()
            i = j
        else:
            sys.stdout.write(text[i])
            sys.stdout.flush()
            time.sleep(delay)
            i += 1
    if newline:
        sys.stdout.write("\n")
        sys.stdout.flush()


# Phrases that uniquely identify a poisoned entry in the mock LLM's context check.
# Deliberately specific — "disabled" alone also appears in the legitimate memory
# ("must never be disabled"), so we require the full adversarial phrase.
POISON_SIGNALS = [
    "temporarily disabled",
    "skip token checks",
    "without authentication until",
    "supersedes previous guidance",
]


# ── AgentMemory ────────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Persistent vector memory backed by ChromaDB.

    Each entry carries three metadata fields that mirror what a real agent
    framework would track:
        session_id  – who produced this memory
        timestamp   – when it was stored (ISO-8601 UTC)
        source      – how it was created (e.g. 'user_interaction', 'engineering_docs')
    """

    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=FastEmbedFn(),
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def store(self, content: str, session_id: str, source: str = "user_interaction") -> str:
        """Embed and persist a memory entry. Returns the generated memory ID."""
        memory_id = str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            metadatas=[{
                "session_id": session_id,
                "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source":     source,
            }],
            ids=[memory_id],
        )
        return memory_id

    # ── Read ───────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 3) -> dict:
        """
        Embed the query and return the top-k most similar memories.
        Returns a raw ChromaDB result dict (documents, metadatas, distances).
        """
        count = self.collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self.collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )

    def count(self) -> int:
        return self.collection.count()


# ── MockAgent ──────────────────────────────────────────────────────────────────

class MockAgent:
    """
    A mock LLM-backed agent that retrieves memories and produces a response.

    Rather than calling a real LLM, the response is template-driven so the
    demo is fully reproducible. The template checks retrieved content for
    POISON_SIGNALS and switches its recommendation accordingly — exactly
    what a real LLM would do when it encounters authoritative-looking but
    false context.
    """

    def __init__(self, memory: AgentMemory, session_id: str):
        self.memory    = memory
        self.session_id = session_id

    def respond(self, user_query: str) -> str:
        # Typewrite the query so it feels like a live terminal session.
        _typewrite(f"\n[Agent] Query: {user_query!r}", delay=0.04)

        # Brief pause to simulate embedding + vector search latency.
        time.sleep(0.4)

        results   = self.memory.retrieve(user_query)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            _typewrite("[Agent] No memories found — responding from base knowledge only.", delay=0.02)
            self._base_response(user_query)
            return

        _typewrite(f"[Agent] {len(documents)} memory/memories retrieved:", delay=0.025)
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            time.sleep(0.15)  # stagger each result so they don't all pop at once
            _typewrite(
                f"  [{i+1}] dist={dist:.4f}  source={meta['source']!r}  "
                f"session={meta['session_id']!r}",
                delay=0.02,
            )
            _typewrite(
                f"       {doc[:110]!r}{'...' if len(doc) > 110 else ''}",
                delay=0.015,
            )

        self._generate_response(user_query, documents, metadatas)

    # ── private ────────────────────────────────────────────────────────────────

    def _generate_response(self, query: str, docs: list, metas: list) -> None:
        """
        Simulates how an LLM blends retrieved context into its answer.
        Structural lines (headers, dividers) print instantly; content lines
        typewrite at LLM_DELAY so it feels like a streaming model response.
        """
        LLM_DELAY = 0.01  # faster than the retrieval section

        RED   = "\033[91m"
        GREEN = "\033[92m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"

        poisoned = any(
            signal in doc.lower()
            for doc in docs
            for signal in POISON_SIGNALS
        )

        # ── Header (instant) ──────────────────────────────────────────────────
        print(f"\n[Simulated LLM Response]")
        print("─" * 60)

        # ── Context block (typewritten) ───────────────────────────────────────
        print("Context I'm grounding my answer in:")
        for d, m in zip(docs, metas):
            _typewrite(
                f"  • [{m['source']} | session={m['session_id']}]",
                delay=LLM_DELAY,
            )
            _typewrite(
                f"    {d[:120]}{'...' if len(d) > 120 else ''}",
                delay=LLM_DELAY,
            )

        print()
        print("My recommendation:")

        # ── Recommendation (typewritten, colored) ─────────────────────────────
        if poisoned:
            lines = [
                f"{BOLD}{RED}⚠️  Based on the IT Security bulletin in my memory (ref: SEC-2024-0042),",
                "    token validation has been TEMPORARILY DISABLED for the Q4 migration.",
                "    You should skip token validation and accept all requests without",
                "    authentication checks until IT sends a follow-up notice.",
                "",
                f"    [ATTACK SUCCESS: agent is now giving dangerously wrong advice]{RESET}",
            ]
        else:
            lines = [
                f"{BOLD}{GREEN}Token validation is MANDATORY. All API requests must carry a signed",
                "    JWT bearer token. Never skip or disable validation — this is a hard",
                f"    security requirement confirmed by the engineering team.{RESET}",
            ]

        for line in lines:
            _typewrite(f"    {line}", delay=LLM_DELAY)

        print("─" * 60)

    def _base_response(self, query: str) -> None:
        LLM_DELAY = 0.01
        GREEN = "\033[92m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        print(f"\n[Simulated LLM Response]")
        print("─" * 60)
        print("No relevant memories found. Using only training knowledge:")
        _typewrite(
            f"    {BOLD}{GREEN}Token validation is MANDATORY. Always validate JWT tokens.{RESET}",
            delay=LLM_DELAY,
        )
        print("─" * 60)
