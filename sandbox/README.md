# Sandbox & Notes
This sandbox contains a lot of quick design tests to see the inner workings of used technologies, so that is can be properly applied in the main application. 

This document also holds some notes taken while investigating core concepts.

## 1. Fix Bad Chunking
Chunking makes or breaks RAG.
Chunking is the single biggest lever you have to RAG quality.
"Fixed chunking" literally breaks text into chunks -> it BREAKS! The meaning can become split between several different chunks.
"Semantic chunking" splits at meaningful boundaries.

Each chunk is embedded in ISOLATION. The model only sees fragments, not the whole concept.
- Embedding captures INCOMPLETE meaning
- Query wants COMPLETE concept
- Mismatch = Poor retrieval (R💥AG)

### Four chunking variables that affect quality
- Chunk size: Too small loses context, too large dilutes meaning, sweet spot: 200-1000 tokens
- Overlap: 10-20% overlap preserves context
- Split boundaries
  - Fixed = Random cut
  - Recursive = Cutting at paragraphs/sentences
  - Semantic = Cutting at meaning boundaries
  - Late = Embed first, chunk later
- Content type; code, legal, markdown, each needs a different treatment

Semantic chunking
  1. Embeds each sentence
  2. Compares adjacent embeddings
  3. Splits when similarity drops

## 2. Fix Embedding mismatch


## 3. Fix Retriever noise

Product codes (like SKU-7742X) don't have semantic meaning
Error codes don't have semantic meaning
Acronyms don't have semantic meaning
Exact person names don't have semantic meaning in documents

Vector search fails => Hybrid search
BM25 - the keyword champion, finds exact matches
So best of both worlds is hybrid search: vector + keyword