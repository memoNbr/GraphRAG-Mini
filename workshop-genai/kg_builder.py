"""
Build the knowledge graph from the three research PDFs.

Usage
-----
    python kg_builder.py               # build any paper not already in the graph
    python kg_builder.py --wipe        # clear the DB, then rebuild from scratch
    python kg_builder.py --file X.pdf  # build one specific file

Why this differs from the original
----------------------------------
1. Embeddings are LOCAL (sentence-transformers, 384-d). No API key, no rate
   limit, free to re-run and re-embed. The LLM is used only for entity and
   relation extraction, so quota is spent in one narrow place.

2. The 1-request-per-second free-tier limit is actually enforced, in two
   layers. The original set max_concurrency=1 believing that respected
   Mistral's limit. It does not: that only limits in-flight requests, so the
   next call fires the instant the previous returns. Calls returning in 400ms
   means 2.5 requests/second and a stream of 429 errors. Worse,
   SimpleKGPipeline takes no entity_relation_extractor= argument at all, so
   the original raised a TypeError at construction and never reached the API.
   Here ThrottledLLM holds the 1 RPS line, and RetryRateLimitHandler backs
   off and retries when the API still says no.

3. The database is NOT wiped by default. The original ran
   "MATCH (n) DETACH DELETE n" on every start, so every run re-extracted every
   chunk and burned the full quota. Worse, a run that died at chunk 30 of 50
   lost those 30 chunks and had to pay for them again. Now a paper already in
   the graph is skipped, so a crashed build resumes instead of restarting.

4. The schema is the real domain schema (schema.py), not the courseware labels
   ["Technology", "Concept", "Cognitive", "Memory"] that returned 0 nodes.

5. All three papers are ingested. The original hardcoded "./data/CoALA.pdf",
   so Simon & Newell and the ACM paper were never read.
"""

import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv


def _load_env():
    """Load .env no matter which folder the script is run from.

    The .env lives at the repo root while these scripts sit one level down in
    workshop-genai/. Plain load_dotenv() searches only from the current
    working directory, so running from the scripts folder silently loads
    NOTHING and every credential comes back as None. Walk up from this file
    instead, so the scripts work regardless of where they are invoked.
    """
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        candidate = folder / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            return candidate
    load_dotenv()
    return None


_load_env()

from neo4j import GraphDatabase
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.utils.rate_limit import RetryRateLimitHandler

from llm_config import build_llm, min_interval_seconds, provider

from schema import (
    DOMAIN_LABELS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GRAPH_SCHEMA,
    LEXICAL_RELATIONSHIPS,
    VECTOR_INDEX_NAME,
)

# --- The three research PDFs ---------------------------------------------
# They live in ./data alongside the Neo4j workshop PDFs, so we select them by
# name rather than by folder. See docs: CoALA (TMLR 2024), Simon & Newell
# (1970), and Rege et al. "Talking with your Car".
PAPERS = [
    "CoALA.pdf",
    "Simon_1970_-_Human_Problem_Solving,_The_State_of_the_Theory_in_1970.pdf",
    "3640792.3675713.pdf",
]

DATA_DIR = Path(__file__).parent / "data"

# 4000 is deliberate. Larger chunks mean fewer extraction calls, which is the
# single most effective way to stay inside a free-tier request budget.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


class ThrottledLLM(LLMInterface):
    """Serialises LLM calls and enforces a minimum interval between them.

    Mistral's free tier allows 1 request per second per key. SimpleKGPipeline
    does not expose the extractor's concurrency and does not accept an
    `entity_relation_extractor=` argument, so the limit has to be enforced at
    the LLM boundary: every call waits its turn, and the next call cannot
    start until the interval has elapsed.

    On its own, `max_concurrency=1` does NOT do this. It only limits how many
    requests are in flight, so the next call fires the instant the previous
    returns. Extraction calls that return in 400ms means 2.5 requests/second
    and a stream of 429 errors.

    This pairs with the RetryRateLimitHandler on the inner client: the
    throttle keeps us under the limit, and the handler backs off and retries
    when the API still says no.

    It also controls structured output. SimpleKGPipeline builds its extractor
    with use_structured_output=llm.supports_structured_output, so reporting
    False here switches extraction to the prompt-based JSON path. That path is
    needed for Groq: its strict schema validator rejects the library's
    Neo4jGraph schema with
        "`additionalProperties:false` must be set on every object"
    because the embedding_properties dict does not set it. Prompt-based
    extraction works with every provider and is what the original workshop
    build used.
    """

    def __init__(
        self,
        llm: LLMInterface,
        min_interval_seconds: float = 1.05,
        use_structured_output: bool = False,
    ):
        self._llm = llm
        self._min_interval = min_interval_seconds
        self._use_structured_output = use_structured_output
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def _reserve(self) -> None:
        """Block until this call may proceed, reserving its slot."""
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if delay > 0:
            time.sleep(delay)

    async def _areserve(self) -> None:
        """Async equivalent, so concurrent coroutines do not block the loop."""
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if delay > 0:
            await asyncio.sleep(delay)

    @property
    def supports_structured_output(self) -> bool:
        # Deliberately NOT delegated. See the class docstring: Groq's schema
        # validator rejects the library's Neo4jGraph schema, so we ask for the
        # prompt-based JSON path instead.
        return self._use_structured_output

    def invoke(self, input, message_history=None, **kwargs):
        self._reserve()
        return self._llm.invoke(input, message_history=message_history, **kwargs)

    async def ainvoke(self, input, message_history=None, **kwargs):
        await self._areserve()
        return await self._llm.ainvoke(
            input, message_history=message_history, **kwargs
        )

    def invoke_with_tools(self, input, tools, message_history=None, **kwargs):
        self._reserve()
        return self._llm.invoke_with_tools(
            input, tools, message_history=message_history, **kwargs
        )

    async def ainvoke_with_tools(
        self, input, tools, message_history=None, **kwargs
    ):
        await self._areserve()
        return await self._llm.ainvoke_with_tools(
            input, tools, message_history=message_history, **kwargs
        )


def _render_schema(schema: dict) -> str:
    """Turn the schema dict into plain text a model can actually use."""
    nodes = [n.get("label") for n in schema.get("node_types", [])]
    rels = [r.get("label") for r in schema.get("relationship_types", [])]
    patterns = schema.get("patterns", [])

    lines = ["NODE TYPES (use these exact labels):"]
    lines.append("  " + ", ".join(nodes))
    lines.append("")
    lines.append("RELATIONSHIP TYPES (use these exact names):")
    lines.append("  " + ", ".join(rels))
    lines.append("")
    lines.append("ALLOWED DIRECTIONS (source -RELATIONSHIP-> target):")
    for pattern in patterns:
        lines.append(
            f"  {pattern.get('source')} -{pattern.get('relationship')}-> "
            f"{pattern.get('target')}"
        )
    return "\n".join(lines)


class ReadableSchemaTemplate(ERExtractionTemplate):
    """Render the schema as plain text instead of a Python dict repr.

    The stock template interpolates schema.model_dump() straight into
    {schema}, so the model is handed a wall of single-quoted Python repr,
    including internal flags it has no use for:

        al_relationship_types': True, 'additional_patterns': True}

    Measured directly against the same model and the same text:
        stock template   -> {"nodes": [], "relationships": []}   (0 entities)
        this template    -> 8-9 entities, correct domain labels
    """

    DEFAULT_TEMPLATE = """
You are extracting a knowledge graph from one chunk of an academic paper.

Return ONLY raw JSON. No prose, no markdown, no code fences, no thinking.

Use exactly this format:
{{"nodes": [{{"id": "0", "label": "Agent", "properties": {{"name": "CoALA"}}}}],
 "relationships": [{{"type": "IMPLEMENTS", "start_node_id": "0",
                    "end_node_id": "1", "properties": {{}}}}]}}

Rules:
- Give every node a unique string id, then reference those ids in relationships.
- "label" MUST be one of the NODE TYPES listed below.
- "type" MUST be one of the RELATIONSHIP TYPES listed below.
- "properties.name" is the entity's name as written in the text.
- Respect the ALLOWED DIRECTIONS; a relationship must point the right way.
- Extract generously: name every architecture, agent, memory or perception
  module, action, concept, theory, method, measurement and vehicle mentioned.
- Reuse an existing node id when the same entity appears again.
- If the chunk truly contains nothing extractable, return
  {{"nodes": [], "relationships": []}}.

Schema:
{schema}

Examples:
{examples}

Input text:
{text}
"""

    def format(self, schema, examples, text=""):
        return super().format(
            text=text,
            schema=_render_schema(schema),
            examples=examples or "",
        )


def connect():
    """Open a Neo4j driver from .env and verify it is reachable."""
    uri, user, pwd = (
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    )
    if not all([uri, user, pwd]):
        sys.exit(
            "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD are not set.\n"
            "Copy .env.example to .env and fill it in."
        )
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    return driver, os.getenv("NEO4J_DATABASE")


def _vector_index_dims(driver, database=None):
    """Return the dimension of VECTOR_INDEX_NAME, or None if absent.

    Verified against the live server (Aura, database 6f1e6cec): SHOW VECTOR
    INDEXES has no top-level 'indexConfig' column. The columns are id, name,
    state, populationPercent, type, entityType, labelsOrTypes, properties,
    indexProvider, owningConstraint, lastRead, readCount, trackedSince,
    options, failureMessage, createStatement. The dimension is nested at
    options['indexConfig']['vector.dimensions'].
    """
    records, _, _ = driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, options RETURN name, options",
        database_=database,
    )
    for record in records:
        if record["name"] == VECTOR_INDEX_NAME:
            options = record["options"] or {}
            index_config = options.get("indexConfig") or {}
            dims = index_config.get("vector.dimensions")
            if dims is not None:
                return int(dims)
    return None


def ensure_vector_index(driver, database=None):
    """Create the vector index at EMBEDDING_DIM, or fail loudly if a stale
    index from the old 1536-d Gemini embedder is still around.

    If the database holds no Chunk nodes, a stale index carries no data, so it
    is simply dropped and recreated. That makes a fresh or empty database work
    with no manual cleanup.
    """
    existing = _vector_index_dims(driver, database)
    if existing is None:
        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=EMBEDDING_DIM,
            similarity_fn="cosine",
            neo4j_database=database,
        )
        print(f"  Vector index '{VECTOR_INDEX_NAME}' created ({EMBEDDING_DIM}-d, cosine)")
        return

    if existing == EMBEDDING_DIM:
        print(f"  Vector index '{VECTOR_INDEX_NAME}' already {EMBEDDING_DIM}-d, keeping it")
        return

    # Dimension mismatch. Safe to replace only if nothing is indexed.
    chunks, _, _ = driver.execute_query(
        "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n",
        database_=database,
    )
    if chunks[0]["n"] == 0:
        drop_index_if_exists(driver, VECTOR_INDEX_NAME, neo4j_database=database)
        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=EMBEDDING_DIM,
            similarity_fn="cosine",
            neo4j_database=database,
        )
        print(
            f"  Vector index was {existing}-d with no data behind it; "
            f"recreated at {EMBEDDING_DIM}-d."
        )
        return

    sys.exit(
        f"Vector index '{VECTOR_INDEX_NAME}' is {existing}-d but the local "
        f"embedder produces {EMBEDDING_DIM}-d.\n"
        f"It was built by the old Gemini embedder, so its vectors are in a\n"
        f"different space and cannot be reused. Run:\n\n"
        f"    python kg_builder.py --wipe\n\n"
        f"to clear the graph (which you want anyway) and rebuild at "
        f"{EMBEDDING_DIM}-d."
    )


def wipe(driver, database=None):
    print("Wiping database...")
    driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)
    drop_index_if_exists(driver, VECTOR_INDEX_NAME, neo4j_database=database)
    print("  Nodes, relationships and vector index cleared.")


def ingested_documents(driver, database=None):
    """Paths of documents already in the graph, so re-runs can skip them."""
    records, _, _ = driver.execute_query(
        "MATCH (d:Document) RETURN d.path AS path", database_=database
    )
    return {record["path"] for record in records}


def count_domain(driver, database=None):
    """(domain node count, typed relationship count) currently in the graph.

    Excludes the lexical graph (Chunk/Document/NEXT_CHUNK/...) and the
    pipeline's internal __KGBuilder__/__Entity__ nodes, so the numbers reflect
    what the schema actually extracted.
    """
    domain, _, _ = driver.execute_query(
        "MATCH (n) "
        "WHERE ANY(l IN labels(n) WHERE l IN $domain_labels) "
        "RETURN count(*) AS c",
        domain_labels=sorted(DOMAIN_LABELS),
        database_=database,
    )
    rels, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN count(r) AS c",
        database_=database,
    )
    lexical, _, _ = driver.execute_query(
        "MATCH ()-[r]->() WHERE type(r) IN $lexical RETURN count(r) AS c",
        lexical=sorted(LEXICAL_RELATIONSHIPS),
        database_=database,
    )
    return domain[0]["c"], rels[0]["c"] - lexical[0]["c"]


def build_pipeline(driver, database):
    # Backoff that is long enough to actually clear a token-window 429.
    #
    # jitter MUST be False here. RetryRateLimitHandler with jitter=True uses
    # tenacity's wait_random_exponential, which draws a RANDOM wait in
    # [min, multiplier * 2**attempt]. Observed live, that meant retries firing
    # after ~2 seconds -- while the server was asking for 21s -- so attempts
    # were burned before the window could refill and the whole build died.
    #
    # With jitter=False we get deterministic 30, 60, 120, 120... which is
    # always longer than the ~21s Groq asks for. A 429 then costs time, not
    # the run.
    rate_limit_handler = RetryRateLimitHandler(
        max_attempts=5,
        min_wait=30.0,
        max_wait=120.0,
        multiplier=2.0,
        jitter=False,
    )

    # Groq / Mistral / Ollama, chosen by LLM_PROVIDER in .env.
    llm = build_llm(rate_limit_handler=rate_limit_handler)
    print(f"  Extraction LLM: {provider()} / {os.getenv('KG_LLM_MODEL') or 'default'}")

    # Enforce the per-provider request interval here. SimpleKGPipeline gives no
    # way to pass a custom extractor or to lower its concurrency, so the LLM
    # boundary is the correct place to hold the line.
    interval = min_interval_seconds()
    # Prompt-based JSON extraction by default. Groq's strict validator rejects
    # the library's Neo4jGraph schema, so structured output is off unless you
    # explicitly opt in with USE_STRUCTURED_OUTPUT=1 (fine for Mistral).
    structured = os.getenv("USE_STRUCTURED_OUTPUT", "0").strip() == "1"
    throttled_llm = ThrottledLLM(
        llm, min_interval_seconds=interval, use_structured_output=structured
    )
    if interval:
        print(f"  Throttle: one request every {interval}s")
    else:
        print("  Throttle: off (local model)")
    print(f"  Structured output: {'on' if structured else 'off (prompt-based JSON)'}")

    # Local, free, no rate limit. The model is downloaded once and cached.
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)

    text_splitter = FixedSizeSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    return SimpleKGPipeline(
        llm=throttled_llm,
        driver=driver,
        embedder=embedder,
        from_file=True,
        text_splitter=text_splitter,
        schema=GRAPH_SCHEMA,
        # The stock template renders the schema as a Python dict repr, which
        # makes the model return zero entities. See ReadableSchemaTemplate.
        prompt_template=ReadableSchemaTemplate(),
        # IGNORE keeps one failed chunk from killing a 30-chunk run and wasting
        # the quota already spent on the first 29.
        on_error="IGNORE",
        neo4j_database=database,
    )


def main():
    parser = argparse.ArgumentParser(description="Build the knowledge graph.")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Clear the database and vector index before building.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Build a single PDF (name inside ./data, or an absolute path).",
    )
    args = parser.parse_args()

    if provider() == "groq" and not os.getenv("GROQ_API_KEY"):
        sys.exit("LLM_PROVIDER=groq but GROQ_API_KEY is not set. Add it to .env.")
    if provider() == "mistral" and not os.getenv("MISTRAL_API_KEY"):
        sys.exit("LLM_PROVIDER=mistral but MISTRAL_API_KEY is not set. Add it to .env.")
    if not os.getenv("NEO4J_URI"):
        sys.exit("NEO4J_URI is not set. Copy .env.example to .env and fill it in.")

    driver, database = connect()
    try:
        if args.wipe:
            wipe(driver, database)

        ensure_vector_index(driver, database)

        if args.file:
            # The help promises "a name inside ./data". Path(args.file) alone
            # resolves against the current working directory, so a bare
            # "CoALA.pdf" would be reported as not found even though it is
            # right there in data/. Resolve against DATA_DIR as a fallback.
            chosen = Path(args.file)
            if not chosen.is_absolute() and not chosen.exists():
                chosen = DATA_DIR / chosen.name
            candidates = [chosen]
        else:
            candidates = [DATA_DIR / name for name in PAPERS]

        missing = [p for p in candidates if not p.exists()]
        if missing:
            for path in missing:
                print(f"  SKIP (not found): {path}")
            if len(missing) == len(candidates):
                sys.exit("No PDFs found. Nothing to build.")
            print()

        already = ingested_documents(driver, database)

        pipeline = build_pipeline(driver, database)

        built = skipped = failed = 0
        for path in candidates:
            if not path.exists():
                skipped += 1
                continue

            # Documents are stored by the path the pipeline was given, so match
            # on the resolved path to stay robust across working directories.
            resolved = str(path.resolve())
            in_graph = any(
                Path(doc).name == path.name for doc in already
            )
            if in_graph:
                print(f"  SKIP (already in graph): {path.name}")
                skipped += 1
                continue

            print(f"\nBuilding {path.name} ...")
            print(
                f"  Running sequentially with backoff. "
                f"Estimated ~1 call per {CHUNK_SIZE} characters."
            )
            try:
                result = asyncio.run(pipeline.run_async(file_path=str(path)))
                print(f"  Done: {result.result}")
            except Exception as exc:
                # A rate limit that outlasted the retries, or a bad chunk, used
                # to propagate out of main() and abandon every other paper.
                # The graph writer runs at the end of a paper, so this paper's
                # work is lost -- but the run as a whole should continue and
                # say so plainly rather than dumping a traceback.
                failed += 1
                print(f"  [FAIL] {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
                print("         Nothing was written for this paper (the writer")
                print("         runs at the end). Re-run to retry it; already")
                print("         built papers are skipped automatically.")
                continue

            # Report what actually landed. With on_error="IGNORE" a bad chunk
            # fails silently, and a build that produces zero entities looks
            # identical to a successful one unless we count.
            domain_nodes, typed_rels = count_domain(driver, database)
            print(f"  -> domain nodes: {domain_nodes}, typed relationships: {typed_rels}")
            if domain_nodes == 0:
                print(
                    "  [FAIL] This paper produced NO domain entities.\n"
                    "         The extraction prompt or model is not working.\n"
                    "         Check that the schema rendered as plain text."
                )
            built += 1

        print("\n" + "=" * 60)
        print(f"Built {built} paper(s), skipped {skipped}, failed {failed}.")
        if failed:
            print("Re-run the same command to retry only what failed.")
        print("Now run:  python agent.py")
        print("(agent.py prints a startup check, so no separate verify step.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
