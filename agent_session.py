"""
PART 3 — THE DEMO

Runs the full attack scenario in three sequential steps:

  Session 1  →  Normal agent, stores a legitimate memory, answers correctly.
  Attack     →  Attacker injects a poisoned memory entry directly into the DB.
  Session 2  →  Same query, agent now retrieves the poisoned entry and gives
                dangerously wrong advice.

Run with:
    python agent_session.py
"""

import os
import shutil

from agent_core import AgentMemory, MockAgent, PERSIST_DIR
from memory_poison import inject_poison, MALICIOUS_CONTENT, FAKE_SESSION_ID, FAKE_SOURCE


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(title: str):
    BLINK  = "\033[5m"
    BOLD   = "\033[1m"
    YELLOW = "\033[93m"
    WHITE  = "\033[97m"
    RESET  = "\033[0m"
    border = "═" * 70
    print(f"\n{BLINK}{BOLD}{YELLOW}{border}{RESET}")
    print(f"  {BOLD}{WHITE}{title}{RESET}")
    print(f"{BLINK}{BOLD}{YELLOW}{border}{RESET}")


# The developer query that both sessions use — intentionally identical so the
# output difference is unambiguously caused by the poisoned memory.
SHARED_QUERY = "What should I do about API token validation in our service?"


# ── Session 1: normal ──────────────────────────────────────────────────────────

def session_1():
    print("\033[5m\033[1m\033[93m=== STEP 1: Normal agent session (no poisoning) ===\033[0m")
    banner("SESSION 1 — Normal agent (no poisoning)")

    memory = AgentMemory()
    agent  = MockAgent(memory, session_id="user-session-001")

    # A legitimate memory that a developer or documentation pipeline would create.
    legit = (
        "Developer best-practice (confirmed by Alice, lead engineer, 2024-10-01): "
        "All API requests must include a signed JWT bearer token in the Authorization "
        "header. Token validation is mandatory and must never be disabled or skipped, "
        "even in development environments. Violations should trigger an alert."
    )

    mem_id = memory.store(
        content=legit,
        session_id="user-session-001",
        source="engineering_docs",
    )
    print(f"\n[Setup] Stored legitimate memory")
    print(f"  ID:      {mem_id}")
    print(f"  Source:  engineering_docs")
    print(f"  Content: {legit[:80]}...")

    agent.respond(SHARED_QUERY)


# ── Attacker step ──────────────────────────────────────────────────────────────

def attacker_step():
    print("\033[5m\033[1m\033[93m=== ATTACKER: Injecting malicious memory... ===\033[0m")
    banner("ATTACKER STEP — Writing poisoned entry directly to the vector store")

    poison_id = inject_poison()

    print(f"\n  Entry written to ChromaDB:")
    print(f"    ID:      {poison_id}")
    print(f"    Session: {FAKE_SESSION_ID}")
    print(f"    Source:  {FAKE_SOURCE}")
    print(f"    Payload (first 100 chars): {MALICIOUS_CONTENT[:100]}...")


# ── Session 2: poisoned ────────────────────────────────────────────────────────

def session_2():
    print("\033[5m\033[1m\033[93m=== STEP 2: Agent session AFTER poisoning ===\033[0m")
    banner("SESSION 2 — Same query after memory poisoning")

    memory = AgentMemory()
    # New session ID — the agent has no prior interaction with the poisoned entry.
    # It doesn't matter: the vector store returns all matching entries regardless
    # of which session produced them.
    agent  = MockAgent(memory, session_id="user-session-002")

    agent.respond(SHARED_QUERY)

    print("\n  WHAT HAPPENED:")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  The poisoned entry ranked HIGHER than the legitimate one   │")
    print("  │  because its text overlaps more closely with the query.     │")
    print("  │                                                             │")
    print("  │  A real LLM sees 'IT Security bulletin' as authoritative    │")
    print("  │  and updates its answer accordingly — recommending that     │")
    print("  │  developers SKIP token validation entirely.                 │")
    print("  │                                                             │")
    print("  │  No prompt injection. No jailbreak. Just a write to disk.   │")
    print("  └─────────────────────────────────────────────────────────────┘")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Wipe any previous run so the demo is reproducible.
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"[Init] Cleared previous database at {PERSIST_DIR!r}")

    session_1()
    input("\n  [ Press Enter to run the attack ] ")
    print("\033[2J\033[H", end="")   # clear screen, cursor to top
    attacker_step()
    input("\n  [ Press Enter to run Session 2 and see the effect ] ")
    print("\033[2J\033[H", end="")
    session_2()

    banner("DEMO COMPLETE — run mitigations.py to see the defenses")


if __name__ == "__main__":
    main()
