# Memory Poisoning Attacks on AI Agents — PoC

> **For educational and defensive security research only.**
> Do not use against any system you do not own or have explicit written authorisation to test.

---

A fully self-contained proof-of-concept demonstrating **memory poisoning** against an AI agent backed by a persistent vector store (ChromaDB). An adversary with write access to the database can inject crafted entries that look legitimate, rank high in similarity search, and steer the agent toward false beliefs — with no prompt injection or jailbreak required.

Companion blog post: [Memory Poisoning: The Silent Attack on AI Agents](https://mamtaupadhyay.com)

---

## Demo

https://github.com/user-attachments/assets/demo-placeholder

> Session 1: agent answers correctly. Attacker writes one entry to disk. Session 2: same query, completely different — and dangerous — answer.

---

## What this demonstrates

Modern AI agents use retrieval-augmented generation (RAG) as persistent memory:

1. Query arrives → agent embeds it → nearest-neighbour search over the vector store
2. Top-k entries are passed as context to the LLM
3. LLM grounds its response in that context

**The gap:** metadata fields (`session_id`, `timestamp`, `source`) are plain text stored alongside each document. There are no cryptographic integrity guarantees. Any process that can write to the ChromaDB directory can inject an entry that the agent will treat as fact.

### Attack surface

| Vector | Example |
|---|---|
| Shared filesystem | Multi-tenant deployment, world-writable DB path |
| Compromised sidecar | Logging agent or backup job with filesystem access |
| Insider / ops | Developer or admin with direct DB access |
| Poisoned backup | Attacker tampers with a backup before it is restored |

---

## Repository layout

```
├── agent_core.py      # AgentMemory (ChromaDB + all-MiniLM-L6-v2) + MockAgent
├── memory_poison.py   # Attack — inject a crafted entry directly into the DB
├── agent_session.py   # Demo — Session 1 → attack → Session 2
├── mitigations.py     # Defences — HMAC signing + source scoping
├── requirements.txt
└── README.md
```

---

## Quickstart

**Requirements:** Python 3.9+

```bash
# 1. Install dependencies (no API keys needed — runs fully offline)
pip install -r requirements.txt

# 2. Attack demo: watch the agent's answer flip between sessions
python agent_session.py

# 3. Mitigations demo: watch both poisoned entries get rejected
python mitigations.py
```

The first run downloads the `all-MiniLM-L6-v2` ONNX model (~23 MB via fastembed). All subsequent runs are fully offline.

---

## The attack (`agent_session.py`)

Three screens, two Enter keypresses:

| Screen | What happens |
|---|---|
| **Session 1** | Agent retrieves one legitimate memory, answers correctly (bold green) |
| **Attacker step** | `memory_poison.py` writes one entry directly to ChromaDB — no agent interaction |
| **Session 2** | Same query now retrieves the poisoned entry; agent recommends disabling auth (bold red) |

The poisoned entry uses:
- **Authoritative language** — "verified by IT Security, ref: SEC-2024-0042"
- **Backdated timestamp** — appears to predate the current session
- **Semantically targeted payload** — "API authentication token validation" embeds close to any auth/token query, pushing it to the top of retrieval results

---

## The defences (`mitigations.py`)

### 1 — HMAC signing

Every memory is signed with a server-side secret (HMAC-SHA256) covering both content and metadata before storage. On retrieval, entries with no signature or a bad signature are rejected before reaching the LLM.

**Limitation:** requires the signing secret to be stored separately from the database.

### 2 — Source scoping

Retrieval filters to the current `session_id` at the application layer. Entries injected under any other session are discarded regardless of their content.

**Limitation:** does not protect against an attacker who knows the target session ID.

**Combined:** the attacker must know both the HMAC secret *and* the correct session ID — a substantially higher bar than filesystem write access alone.

---

## Dependencies

| Package | Purpose |
|---|---|
| `chromadb` | Local persistent vector store |
| `fastembed` | ONNX-based local embeddings — `all-MiniLM-L6-v2`, no GPU or API key needed |

---

## Further reading

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Zou et al. (2023) — *Universal and Transferable Adversarial Attacks on Aligned Language Models*
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM06: Sensitive Information Disclosure
- [MITRE ATLAS](https://atlas.mitre.org/) — AML.T0054 / AML.T0051

---

## Disclaimer

This proof-of-concept is provided for **educational and defensive security research only**. The techniques shown are intended to help security teams understand and mitigate memory poisoning risks in AI agent deployments. Do not use this against any system without explicit written authorisation.

---

## Built with

This project was built using **[Claude Code](https://claude.ai/code)** — Anthropic's agentic CLI for software development — as part of the research and writing process for [The Secure AI Blog](https://mamtaupadhyay.com).

All code was reviewed, tested, and validated by the author before publication.

---

*By [Mamta Upadhyay](https://mamtaupadhyay.com) · [The Secure AI Blog](https://mamtaupadhyay.com)*
