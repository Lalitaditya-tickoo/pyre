"""The fixed prompt suite.

Frozen on purpose. Every number in RESULTS.md is produced against these exact
strings, so a week-8 result is comparable to a week-1 result. If you ever need
to change them, add a new suite with a new name rather than editing this one —
otherwise the whole results table silently stops meaning anything.
"""

SHORT = [
    "Explain what a hash table is.",
    "Write a Python function that reverses a linked list.",
    "What causes sea breezes?",
    "Summarise the difference between TCP and UDP.",
    "Give three uses for a paperclip.",
    "What is the time complexity of quicksort in the worst case?",
    "Describe the CAP theorem in two sentences.",
    "How does a diesel engine differ from a petrol engine?",
]

# Longer prefill, to separate prefill-bound from decode-bound behaviour.
LONG = [
    (
        "You are reviewing a pull request. The diff adds a caching layer in front of a "
        "database query path, keyed by user id, with a five minute TTL and no explicit "
        "invalidation on write. The service runs eight replicas behind a load balancer "
        "and the write path is handled by a separate service that does not know the "
        "cache exists. Walk through the correctness problems with this design, in order "
        "of how likely they are to be hit in production, and say what you would change. "
        "Question: what is the single most serious issue here?"
    ),
    (
        "A distributed job scheduler assigns tasks to workers using a lease with a thirty "
        "second timeout. Workers renew leases every ten seconds. If a worker is partitioned "
        "from the coordinator but keeps running, the coordinator will reassign its task "
        "after the lease expires, so two workers may run the same task concurrently. The "
        "task writes to object storage under a deterministic key. Explain what guarantees "
        "this system does and does not provide, and how you would make the write path safe."
    ),
]

# Shared prefix pair — the workload PYRE is meant to win on once the radix
# prefix cache lands. Both entries share a long identical system preamble.
_PREFIX = (
    "You are a careful senior engineer. Answer precisely, admit uncertainty, prefer "
    "concrete detail over generality, and never invent API names. When a question is "
    "ambiguous, state the interpretation you chose before answering it. Keep answers "
    "under two hundred words unless the question demands more. "
)
SHARED_PREFIX = [_PREFIX + q for q in SHORT]

SUITES = {"short": SHORT, "long": LONG, "shared_prefix": SHARED_PREFIX}


def get_suite(name: str, batch_size: int) -> list[str]:
    """Return exactly ``batch_size`` prompts, cycling the suite if needed."""
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; have {sorted(SUITES)}")
    base = SUITES[name]
    return [base[i % len(base)] for i in range(batch_size)]
