"""
Domain schema for the three research papers.

The previous schema was ["Technology", "Concept", "Cognitive", "Memory"].
All four labels returned 0 nodes: they describe GenAI courseware, not the
papers. These types are drawn from what the three papers actually discuss.

  CoALA (Sumers et al., TMLR 2024)   - cognitive architectures for language agents
  Simon & Newell (1970)              - Human Problem Solving, the state of the theory
  Rege et al.                        - "Talking with your Car", conversational AI in AVs

Shared thread: a cognitive architecture, its memory and action modules, and how
that architecture is applied to a real interface (an autonomous vehicle).
"""

# Node types the extractor is guided toward.
NODE_TYPES = [
    "Agent",
    "Architecture",
    "Memory",
    "Perception",
    "Action",
    "Vehicle",
    "Concept",
    "Theory",
    "Method",
    "Measurement",
]

# Relationship types the extractor is guided toward.
RELATIONSHIP_TYPES = [
    "IMPLEMENTS",
    "EXTENDS",
    "PROVIDES",
    "APPLIES_TO",
    "MEASURES",
    "CONTRADICTS",
    "CITES",
]

# Allowed (source, relationship, target) triples.
# This is what makes the schema useful: it encodes how the types connect, so
# the graph shows the relationship between the papers rather than a flat list.
PATTERNS = [
    ("Architecture", "IMPLEMENTS", "Agent"),
    ("Agent", "IMPLEMENTS", "Memory"),
    ("Agent", "IMPLEMENTS", "Perception"),
    ("Agent", "IMPLEMENTS", "Action"),
    ("Architecture", "EXTENDS", "Architecture"),
    ("Theory", "EXTENDS", "Theory"),
    ("Method", "PROVIDES", "Theory"),
    ("Concept", "PROVIDES", "Method"),
    ("Memory", "PROVIDES", "Concept"),
    ("Theory", "APPLIES_TO", "Agent"),
    ("Method", "APPLIES_TO", "Vehicle"),
    ("Agent", "APPLIES_TO", "Vehicle"),
    ("Concept", "APPLIES_TO", "Agent"),
    ("Measurement", "MEASURES", "Agent"),
    ("Measurement", "MEASURES", "Concept"),
    ("Theory", "CONTRADICTS", "Theory"),
    ("Method", "CITES", "Method"),
]

# The dict handed to SimpleKGPipeline(schema=...).
# Keys are node_types / relationship_types / patterns (NOT "relation_types").
#
# additional_* are True on purpose. A too-strict schema is what produced an
# empty graph last time: anything the extractor saw that did not fit the label
# list was silently dropped. These act as a safety valve so a real concept
# still lands in the graph instead of vanishing.
GRAPH_SCHEMA = {
    "node_types": NODE_TYPES,
    "relationship_types": RELATIONSHIP_TYPES,
    "patterns": PATTERNS,
    "additional_node_types": True,
    "additional_relationship_types": True,
    "additional_patterns": True,
}

# Name of the vector index over Chunk.embedding.
# Both kg_builder.py (create) and agent.py (query) must use this same name.
VECTOR_INDEX_NAME = "chunkEmbedding"

# The lexical graph SimpleKGPipeline always creates for chunk/document
# bookkeeping. Not part of the domain schema, so excluded from quality counts.
LEXICAL_RELATIONSHIPS = {"FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"}
LEXICAL_NODE_LABELS = {"Chunk", "Document", "__KGBuilder__", "__Entity__"}

# The node labels the schema is supposed to produce.
DOMAIN_LABELS = set(NODE_TYPES)

# Must match the embedding model used by SentenceTransformerEmbeddings.
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
