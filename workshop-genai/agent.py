"""
Interactive chat over the knowledge graph.

Cost per question: one LOCAL embedding (free, no API) plus however many chat
LLM calls the agent loop decides it needs. Nothing else is touched. There is
no per-query embedding quota to exhaust, which was half the rate-limit pain.

Usage
-----
    python agent.py
    python agent.py --model mistral-small-latest

Design notes
------------
* One provider. kg_builder.py embeds and extracts with Mistral; this chats
  with Mistral. The original built the graph with mistral-embed and queried it
  with gemini-embedding-001. Embeddings from two different models are NOT
  comparable, so that vector search was returning near-random documents.

* Most questions should not need vector search at all. The agent has a
  read-only Cypher tool and a describe-entity tool, both of which are free and
  deterministic. Vector search is the fallback, not the default.
"""

import argparse
import os
import re
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

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever

from llm_config import chat_model_kwargs, provider

from schema import EMBEDDING_DIM, EMBEDDING_MODEL, VECTOR_INDEX_NAME

# Overridden by --model, and defaults to whatever LLM_PROVIDER selects.
DEFAULT_MODEL = None

# For the startup check: lexical scaffolding the pipeline adds automatically,
# versus the domain types the schema is supposed to produce.
SCAFFOLD_LABELS = {"__KGBuilder__", "__Entity__"}
DOMAIN_LABELS = {
    "Agent", "Architecture", "Memory", "Perception", "Action",
    "Vehicle", "Concept", "Theory", "Method", "Measurement",
}

# Anything that would write. The Cypher tool is read-only by construction.
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|GRANT|DENY|REVOKE|"
    r"LOAD\s+CSV|FOREACH|ALTER|START)\b",
    re.IGNORECASE,
)

# Pull the chunk text plus a readable summary of the entities around it, so
# one retrieval call yields both the prose and the graph structure.
RETRIEVAL_QUERY = """
MATCH (node)-[:FROM_DOCUMENT]->(d)
RETURN
    node.text AS text, score,
    d.path AS document_source,
    collect {
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            [l IN labels(entity)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            entity.name,
            type(r),
            [l IN labels(other)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            other.name
            ]) AS values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } AS associated_entities
"""


def connect():
    uri, user, pwd = (
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    )
    if not all([uri, user, pwd]):
        raise SystemExit(
            "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD are not set.\n"
            "Copy .env.example to .env and fill it in."
        )
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    return driver, os.getenv("NEO4J_DATABASE")


def _columns(keys, records):
    """Column names, falling back to the first record's keys.

    execute_query always returns keys, but deriving them from the records
    costs nothing and keeps this from exploding on an unusual driver.
    """
    if keys:
        return list(keys)
    if records:
        return list(records[0].keys())
    return []


def _rows_to_text(records, columns, limit=40):
    if not records:
        return "(no rows)"
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))
    for record in records[:limit]:
        cells = []
        for column in columns:
            value = record[column]
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            cells.append("" if value is None else str(value))
        lines.append(" | ".join(cells))
    if len(records) > limit:
        lines.append(f"... {len(records) - limit} more rows")
    return "\n".join(lines)


def make_tools(driver, database, retriever):
    @tool("get-graph-schema")
    def get_schema() -> str:
        """List the node labels and relationship types present in the graph."""
        labels, _, _ = driver.execute_query(
            "CALL db.labels() YIELD label "
            "WHERE NOT label IN ['__KGBuilder__', '__Entity__'] "
            "RETURN label ORDER BY label",
            database_=database,
        )
        rels, _, _ = driver.execute_query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "WHERE NOT relationshipType IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK'] "
            "RETURN relationshipType ORDER BY relationshipType",
            database_=database,
        )
        counts, _, _ = driver.execute_query(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS nodes ORDER BY nodes DESC",
            database_=database,
        )
        return (
            "Node labels:\n"
            + _rows_to_text(counts, ["label", "nodes"])
            + "\n\nRelationship types:\n"
            + _rows_to_text(rels, ["relationshipType"])
        )

    @tool("query-graph")
    def query_graph(cypher: str) -> str:
        """Run a READ-ONLY Cypher query against the knowledge graph.

        Use this for anything structural: counting nodes by type, listing
        entities of a kind, walking relationships, comparing two papers.
        Prefer this over vector search when the question is about the graph
        itself rather than about prose in the PDFs.
        """
        if _WRITE_KEYWORDS.search(cypher):
            return "Refused: this tool is read-only. No CREATE/MERGE/SET/DELETE."
        try:
            records, _, keys = driver.execute_query(
                cypher, database_=database
            )
        except Exception as exc:
            return f"Cypher error: {exc}"
        return _rows_to_text(records, _columns(keys, records))

    @tool("describe-entity")
    def describe_entity(name: str) -> str:
        """Show one entity by name and everything directly connected to it.

        Best tool for 'what is X' or 'how does X relate to Y' when X is a
        named concept, architecture, module, theory or vehicle.
        """
        query = """
        MATCH (n) WHERE toLower(n.name) = toLower($name)
        OPTIONAL MATCH (n)-[r]->(m)
        OPTIONAL MATCH (p)-[r2]->(n)
        RETURN labels(n)[0] AS type, n.name AS name,
               collect(DISTINCT type(r) + ' -> ' + coalesce(m.name, m.text)) AS outgoing,
               collect(DISTINCT type(r2) + ' <- ' + coalesce(p.name, p.text)) AS incoming
        LIMIT 1
        """
        try:
            records, _, keys = driver.execute_query(
                query, name=name, database_=database
            )
        except Exception as exc:
            return f"Cypher error: {exc}"
        if not records or records[0]["name"] is None:
            return f"No entity named '{name}'. Try get-graph-schema or query-graph."
        return _rows_to_text(records, _columns(keys, records))

    @tool("search-documents")
    def search_documents(query: str) -> str:
        """Semantic search across the text of the ingested PDFs.

        Use this when the question is about what a paper actually SAYS
        (an argument, a definition, a finding, a quote). Free to call: the
        query is embedded locally, not through an API.
        """
        try:
            result = retriever.search(query_text=query, top_k=8)
        except Exception as exc:
            return f"Retrieval error: {exc}"
        if not result.items:
            return "(no matching passages)"
        chunks = []
        for item in result.items:
            meta = item.metadata or {}
            source = meta.get("document_source", "unknown")
            entities = meta.get("associated_entities") or []
            entities = [e for e in entities if e and e.strip()]
            header = f"[{source} | score={meta.get('score', '?'):.3f}]"
            body = item.content or ""
            related = (
                "\n  related: " + "; ".join(entities[:12]) if entities else ""
            )
            chunks.append(f"{header}\n{body}{related}")
        return "\n\n---\n\n".join(chunks)

    return [get_schema, query_graph, describe_entity, search_documents]


def health_check(driver, database):
    """Report whether the graph is ready, then return True if usable.

    This replaces a separate inspection script: 'python agent.py' is the one
    command. It prints exactly what is wrong when something is wrong, instead
    of failing later with a confusing retrieval error.
    """
    print("=" * 62)
    print("STARTUP CHECK")
    print("=" * 62)
    ready = True

    # Credentials. Values are never printed.
    env_file = _load_env()
    which = provider()
    key_name = {"groq": "GROQ_API_KEY", "mistral": "MISTRAL_API_KEY"}.get(which)
    print(f"  .env              : {env_file or 'NOT FOUND'}")
    print(f"  LLM_PROVIDER      : {which}")
    if key_name:
        print(f"  {key_name + ' ' * (18 - len(key_name))}: {'set' if os.getenv(key_name) else 'MISSING'}")
    else:
        print(f"  LLM_PROVIDER      : {which} (local, no API key needed)")
    print(f"  NEO4J_URI         : {'set' if os.getenv('NEO4J_URI') else 'MISSING'}")

    # Documents.
    try:
        docs, _, _ = driver.execute_query(
            "MATCH (d:Document) RETURN d.path AS path ORDER BY path",
            database_=database,
        )
    except Exception as exc:
        print(f"  [FAIL] Cannot query Neo4j: {exc}")
        return False

    print(f"  documents         : {len(docs)}")
    for record in docs:
        print(f"      - {os.path.basename(record['path'])}")
    if not docs:
        print("  [FAIL] No documents. Run:  python kg_builder.py --wipe")
        ready = False

    # Node mix. The primary signal is domain nodes and typed relationships.
    # __KGBuilder__ / __Entity__ are structural nodes the pipeline always
    # creates for entity resolution and chunk linking, so their share is
    # informational, not a quality judgement. A real schema failure looks like
    # zero domain nodes and zero typed relationships.
    labels, _, _ = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS nodes ORDER BY nodes DESC",
        database_=database,
    )
    total = sum(r["nodes"] for r in labels) or 1
    scaffolding = sum(r["nodes"] for r in labels if r["label"] in SCAFFOLD_LABELS)
    domain = [(r["label"], r["nodes"]) for r in labels if r["label"] in DOMAIN_LABELS]
    domain_total = sum(count for _, count in domain)

    rels, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC",
        database_=database,
    )
    lexical = {"FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"}
    typed = [(r["t"], r["n"]) for r in rels if r["t"] not in lexical]
    typed_total = sum(count for _, count in typed)

    print(f"  domain nodes      : {domain_total}")
    if domain:
        shown = ", ".join(f"{name} {count}" for name, count in domain[:8])
        print(f"                      {shown}")
    print(f"  typed relations   : {typed_total}")
    if typed:
        shown = ", ".join(f"{name} {count}" for name, count in typed[:8])
        print(f"                      {shown}")
    print(f"  pipeline internals: {scaffolding} (__KGBuilder__/__Entity__)")

    if domain_total == 0:
        print("  [FAIL] The domain schema never applied. Re-run kg_builder.py.")
        ready = False
    if typed_total == 0:
        print("  [FAIL] No typed relationships. Check the extraction prompt.")
        ready = False

    # Vector index dimension must match the local embedder.
    try:
        indexes, _, _ = driver.execute_query(
            "SHOW VECTOR INDEXES YIELD name, options RETURN name, options",
            database_=database,
        )
        dims = None
        for record in indexes:
            if record["name"] == VECTOR_INDEX_NAME:
                options = record["options"] or {}
                dims = (options.get("indexConfig") or {}).get("vector.dimensions")
        if dims is None:
            print(f"  [FAIL] Vector index '{VECTOR_INDEX_NAME}' missing.")
            ready = False
        elif str(dims) != str(EMBEDDING_DIM):
            print(
                f"  [FAIL] Index is {dims}-d but the local embedder is "
                f"{EMBEDDING_DIM}-d. Run: python kg_builder.py --wipe"
            )
            ready = False
        else:
            print(f"  vector index      : '{VECTOR_INDEX_NAME}' {dims}-d, matches embedder")
    except Exception as exc:
        print(f"  [FAIL] Cannot read indexes: {exc}")
        ready = False

    print("-" * 62)
    print("RESULT:", "READY" if ready else "NOT READY - fix the [FAIL] items above")
    print("=" * 62)
    print()
    return ready


def main():
    parser = argparse.ArgumentParser(description="Chat with the knowledge graph.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    driver, database = connect()

    if not health_check(driver, database):
        driver.close()
        raise SystemExit(1)

    # Local embedding: free, no rate limit, and it matches what kg_builder.py
    # wrote into the index (all-MiniLM-L6-v2, 384-d).
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)

    retriever = VectorCypherRetriever(
        driver,
        neo4j_database=database,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
    )

    model_name, model_kwargs = chat_model_kwargs()
    if args.model:
        model_name = args.model
    model = init_chat_model(model_name, **model_kwargs)
    agent = create_agent(model, make_tools(driver, database, retriever))

    print(f"Agent ready ({provider()}: {model_name}). Type 'exit' to quit.")
    print("Tools: get-graph-schema, query-graph, describe-entity, search-documents\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_query.lower() in ("exit", "quit"):
            break
        if not user_query:
            continue

        print()
        for step in agent.stream(
            {"messages": [{"role": "user", "content": user_query}]},
            stream_mode="values",
        ):
            last = step["messages"][-1]
            if last.type == "ai":
                last.pretty_print()
        print("\n" + "-" * 60 + "\n")

    driver.close()


if __name__ == "__main__":
    main()
