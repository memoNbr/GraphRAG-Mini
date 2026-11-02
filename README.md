# graphrag-pdf

Turn a folder of academic PDFs into a knowledge graph you can actually talk to.

Chunk the papers, extract typed entities and relationships with an LLM, embed the
chunks locally, store everything in Neo4j, then chat with an agent that has read-only
tools over the graph.

![Graph with all ten domain node types](workshop-genai/data/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202026-11-02%20160502.png)

![Agent startup health check reporting a READY graph](workshop-genai/data/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202026-11-02%20160610.png)

## The papers

| File | Work |
|---|---|
| `CoALA.pdf` | Sumers et al., *A Cognitive Architecture for Language Agents* (TMLR 02/2024) |
| `Simon_1970_...pdf` | Simon & Newell, *Human Problem Solving: The State of the Theory in 1970* |
| `3640792.3675713.pdf` | Rege, Currano, Sirkin & Kim, *Talking with your Car* |

These PDFs are **not redistributed** — they are gitignored. Drop your own into
`workshop-genai/data/`.

## Quick start

```bash
pip install -r requirements.txt
pip install sentence-transformers pypdf python-dotenv

cp .env.example .env                     # then fill in your keys
```

```bash
cd workshop-genai
python kg_builder.py --file CoALA.pdf    # build the graph
python agent.py                          # chat with it
```

`agent.py` runs a health check at startup, so there is no separate verify step.

## What you need in `.env`

| Key | Required | Notes |
|---|---|---|
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | yes | Neo4j Aura or local |
| `GROQ_API_KEY` | yes (default) | free tier, generous |
| `MISTRAL_API_KEY` | only if `LLM_PROVIDER=mistral` | free tier exhausts quickly |

**Embeddings need no key at all.** They run locally on
`sentence-transformers/all-MiniLM-L6-v2` (384-d), so re-embedding is free and
never touches a rate limit.

## Switching the LLM

Set `LLM_PROVIDER` in `.env` — one line, no code changes:

```ini
LLM_PROVIDER=groq      # default, verified working
LLM_PROVIDER=mistral   # free tier is easy to exhaust (429 code 1300)
LLM_PROVIDER=ollama    # fully local; needs `ollama serve` + pip install ollama
```

`llm_config.py` is the single place provider details live.

## The domain schema

Ten node types, seven relationship types, seventeen allowed patterns — defined in
`schema.py` and drawn from what the papers actually discuss:

```
Agent · Architecture · Memory · Perception · Action
Vehicle · Concept · Theory · Method · Measurement

IMPLEMENTS · EXTENDS · PROVIDES · APPLIES_TO · MEASURES · CONTRADICTS · CITES
```

`additional_*` are all `True` on purpose. A too-strict schema silently drops
anything that does not fit the label list, which is how you end up with an
empty graph.

## Rate limits: read this before you change anything

The Groq free tier is capped at **8000 tokens per minute** on every usable chat
model — not requests. Read it live from `x-ratelimit-limit-tokens`.

Measured on real CoALA chunks: ~1500 prompt + ~400–1500 completion = **~1900–3500
tokens per chunk**. Two consequences:

1. **Requests must be ~30 seconds apart.** `LLM_MIN_INTERVAL` defaults to `30.0`
   for Groq. An earlier value of `2.0` was derived from a requests/minute
   assumption and was ~15× too fast — that produced a wall of 429s.
2. **Retry backoff must not use jitter.** With `jitter=True`,
   `RetryRateLimitHandler` draws a *random* wait, so retries fired after ~2
   seconds while the server was asking for 21s, burning every attempt before the
   window could refill. It is now `jitter=False` → deterministic 30, 60, 120.

Also worth knowing: **`reasoning_tokens` count against TPM.** A reasoning model
can spend its whole output budget thinking and return nothing.

At 30s intervals, `CoALA.pdf` (31 chunks) takes ~16 minutes.

## Two things that silently produce zero entities

Both were found by running against the real API, not by tests:

1. **Groq rejects the library's structured-output schema.** Strict mode demands
   `additionalProperties: false` on every object; `neo4j-graphrag`'s `Neo4jGraph`
   does not set it on `embedding_properties`, so you get a hard `400`.
   `ThrottledLLM` therefore reports `supports_structured_output = False` and the
   extractor falls back to prompt-based JSON, which works everywhere.

2. **The stock prompt renders the schema as a Python dict repr.** The model is
   handed a wall of single-quoted Python including internal flags:

   ```
   al_relationship_types': True, 'additional_patterns': True}
   ```

   Same model, same text, measured directly:

   | Prompt | Result |
   |---|---|
   | stock `ERExtractionTemplate` | `{"nodes": [], "relationships": []}` — **0 entities** |
   | `ReadableSchemaTemplate` | 13 nodes / 10 relationships |

   `ReadableSchemaTemplate` renders node types, relationship types and allowed
   directions as plain text.

## Agent tools

All read-only:

| Tool | Purpose |
|---|---|
| `get-graph-schema` | node and relationship types |
| `query-graph` | Cypher, with a write-keyword guard |
| `describe-entity` | one entity and its neighbourhood |
| `search-documents` | vector search over chunks |

## Project layout

```
.env.example          documents every key (copy to .env)
requirements.txt      direct dependencies only
README.md

workshop-genai/
  kg_builder.py       PDF -> knowledge graph (single command)
  agent.py            interactive chat + startup health check
  llm_config.py       provider switch: groq | mistral | ollama
  schema.py           domain schema, index name, embedding dims
  data/               your PDFs (gitignored) + screenshots
```
