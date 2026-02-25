#!/usr/bin/env python3
"""
Generate realistic 10-layer LLM architecture data for the Deconstruction visualizer.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_layer_data.py "Plan a 3-day trip to Tokyo"
    python generate_layer_data.py "Explain quantum computing to a 10-year-old"

Outputs a JSON file in the output/ directory.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import tiktoken
from openai import OpenAI


# --- Configuration ---

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
ENCODING_NAME = "cl100k_base"

# A small "knowledge base" for semantic search simulation.
# Customize these to match the domains you want to demonstrate.
KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "text": "User previously mentioned: likes museums and traditional food",
        "category": "user_preferences",
    },
    {
        "id": "doc_02",
        "text": "Travel preferences: morning activities, cultural sites, local experiences",
        "category": "user_preferences",
    },
    {
        "id": "doc_03",
        "text": "Previous trip to Kyoto, enjoyed temples and tea ceremonies",
        "category": "travel_history",
    },
    {
        "id": "doc_04",
        "text": "Budget range: moderate, prefers quality over luxury",
        "category": "user_preferences",
    },
    {
        "id": "doc_05",
        "text": "Timezone: GMT+1, typically plans trips 2-3 months ahead",
        "category": "user_metadata",
    },
    {
        "id": "doc_06",
        "text": "Interested in technology, AI, and science topics",
        "category": "user_interests",
    },
    {
        "id": "doc_07",
        "text": "Prefers structured, step-by-step explanations with examples",
        "category": "communication_style",
    },
    {
        "id": "doc_08",
        "text": "Has background in software engineering and data science",
        "category": "user_background",
    },
    {
        "id": "doc_09",
        "text": "Previously asked about machine learning fundamentals and neural networks",
        "category": "query_history",
    },
    {
        "id": "doc_10",
        "text": "Enjoys cooking, especially Japanese and Mediterranean cuisine",
        "category": "user_interests",
    },
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def generate_layer_data(prompt: str) -> dict:
    client = OpenAI()
    encoder = tiktoken.get_encoding(ENCODING_NAME)

    print(f"Prompt: {prompt}\n")

    # =========================================================================
    # Layer 1: Raw Input
    # =========================================================================
    print("Layer 1: Raw Input")
    layer1 = {
        "user_message": prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": None,
    }

    # =========================================================================
    # Layer 2: Tokenization (Encode)
    # =========================================================================
    print("Layer 2: Tokenization")
    token_ids = encoder.encode(prompt)
    # Decode each token individually to get the string representation
    tokens = [encoder.decode([tid]) for tid in token_ids]

    layer2 = {
        "input_text": prompt,
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": len(token_ids),
        "encoding_method": f"tiktoken ({ENCODING_NAME})",
        "vocab_size": encoder.n_vocab,
        "code": (
            f"import tiktoken\n"
            f"\n"
            f'encoder = tiktoken.get_encoding("{ENCODING_NAME}")\n'
            f'text = "{prompt}"\n'
            f"token_ids = encoder.encode(text)\n"
            f"# Output: {token_ids}"
        ),
    }

    # =========================================================================
    # Layer 3: Embeddings
    # =========================================================================
    print("Layer 3: Embeddings (calling OpenAI API)")

    # Embed the full prompt
    prompt_embedding_resp = client.embeddings.create(
        model=EMBEDDING_MODEL, input=prompt
    )
    prompt_embedding = prompt_embedding_resp.data[0].embedding
    embedding_dim = len(prompt_embedding)

    # Embed individual tokens for visualization
    token_embeddings = {}
    if len(tokens) <= 20:  # Only embed individual tokens if reasonable count
        token_embed_resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=tokens
        )
        for i, tok in enumerate(tokens):
            vec = token_embed_resp.data[i].embedding
            label = f"token_{token_ids[i]}_{tok.strip() or repr(tok)}"
            # Store first 5 values + indicator of remaining
            token_embeddings[label] = vec[:5] + [f"...({embedding_dim - 5} more)"]

    layer3 = {
        "token_ids": token_ids,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": embedding_dim,
        "shape": [len(token_ids), embedding_dim],
        "prompt_embedding_sample": prompt_embedding[:10]
        + [f"...({embedding_dim - 10} more)"],
        "token_embeddings": token_embeddings,
        "explanation": (
            f"Each token ID is converted to a {embedding_dim}-dimensional vector "
            f"that captures semantic meaning"
        ),
        "code": (
            f"embeddings = model.embed_tokens(token_ids)\n"
            f"# Shape: [{len(token_ids)} tokens, {embedding_dim} dimensions]\n"
            f"# Each token becomes a dense vector representation"
        ),
    }

    # =========================================================================
    # Layer 4: Semantic Search (Vector Similarity)
    # =========================================================================
    print("Layer 4: Semantic Search (embedding knowledge base)")

    # Embed all knowledge base documents
    kb_texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    kb_embed_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=kb_texts)
    kb_embeddings = [item.embedding for item in kb_embed_resp.data]

    # Compute cosine similarity between prompt and each document
    prompt_vec = np.array(prompt_embedding)
    similarities = []
    for i, doc in enumerate(KNOWLEDGE_BASE):
        doc_vec = np.array(kb_embeddings[i])
        sim = cosine_similarity(prompt_vec, doc_vec)
        similarities.append(
            {
                "id": doc["id"],
                "category": doc["category"],
                "similarity": round(sim, 4),
                "snippet": doc["text"],
            }
        )

    # Sort by similarity descending, take top 5
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:5]

    layer4 = {
        "query_embedding": f"embedding_of('{prompt[:40]}...')"
        if len(prompt) > 40
        else f"embedding_of('{prompt}')",
        "search_method": "cosine_similarity",
        "top_k": 5,
        "total_docs_searched": len(KNOWLEDGE_BASE),
        "retrieved_docs": top_results,
        "all_similarities": similarities,  # Full list for visualization
        "code": (
            "from numpy import dot\n"
            "from numpy.linalg import norm\n"
            "\n"
            "def cosine_similarity(a, b):\n"
            "    return dot(a, b) / (norm(a) * norm(b))\n"
            "\n"
            f'query_vec = embed("{prompt[:30]}...")\n'
            "results = []\n"
            "for doc in vector_db:\n"
            "    sim = cosine_similarity(query_vec, doc.embedding)\n"
            "    results.append((doc, sim))\n"
            "\n"
            "top_5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]"
        ),
    }

    # =========================================================================
    # Layer 5: Context Injection (MCP) — illustrative
    # =========================================================================
    print("Layer 5: Context Injection (illustrative)")

    layer5 = {
        "retrieved_context": [doc["snippet"] for doc in top_results],
        "injected_into_prompt": True,
        "source": "Model Context Protocol (MCP)",
        "mcp_structure": {
            "user_input": prompt,
            "system_state": "conversation_turn_1",
            "memory_objects": list(
                set(doc["category"] for doc in top_results)
            ),
            "tool_use": None,
            "context_graph": " → ".join(
                [doc["category"] for doc in top_results[:3]] + ["response_generation"]
            ),
        },
        "code": (
            "context = {\n"
            '    "user_input": user_message,\n'
            '    "memory": retrieve_from_vector_db(query_embedding),\n'
            '    "system_state": session.state,\n'
            '    "timestamp": session.timestamp\n'
            "}\n"
            "enriched_prompt = assemble_prompt(context)"
        ),
    }

    # =========================================================================
    # Layer 6: Task Decomposition (LangChain) — illustrative
    # =========================================================================
    print("Layer 6: Task Decomposition (illustrative)")

    # Use GPT to generate plausible task decomposition steps for this prompt
    decomp_resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a user prompt, output 4-6 task decomposition steps that "
                    "a LangChain SequentialChain would use to process it. "
                    "Output ONLY a JSON array of short snake_case step names. "
                    'Example: ["understand_intent", "retrieve_data", "generate_plan", "format_output"]'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    try:
        steps = json.loads(decomp_resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        steps = [
            "understand_user_intent",
            "retrieve_relevant_information",
            "generate_structured_response",
            "format_output",
        ]

    layer6 = {
        "orchestrator": "LangChain",
        "chain_type": "SequentialChain",
        "steps": steps,
        "code": (
            "from langchain.chains import SequentialChain\n"
            "\n"
            "tasks = [\n"
            + "".join(
                f'    ("{step}", {step.title().replace("_", "")}Chain()),\n'
                for step in steps
            )
            + "]\n"
            "\n"
            "chain = SequentialChain(chains=tasks)\n"
            f'result = chain.run(user_input=enriched_prompt)'
        ),
    }

    # =========================================================================
    # Layer 7: Attention Mechanism — illustrative
    # =========================================================================
    print("Layer 7: Attention Mechanism (illustrative)")

    # Generate a plausible attention matrix for the input tokens
    n_tokens = len(tokens)
    # Create a random but structured attention pattern:
    # - Causal mask (lower triangular)
    # - Higher weights for adjacent tokens and semantically related ones
    np.random.seed(sum(token_ids) % 2**31)
    raw_attention = np.random.rand(n_tokens, n_tokens)
    # Apply causal mask
    causal_mask = np.tril(np.ones((n_tokens, n_tokens)))
    raw_attention *= causal_mask
    # Boost diagonal and adjacent
    raw_attention += np.eye(n_tokens) * 0.5
    for i in range(n_tokens):
        if i > 0:
            raw_attention[i][i - 1] += 0.3
    # Normalize rows to sum to 1 (softmax-like)
    row_sums = raw_attention.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attention_matrix = (raw_attention / row_sums).tolist()
    # Round for readability
    attention_matrix = [[round(v, 4) for v in row] for row in attention_matrix]

    layer7 = {
        "mechanism": "Multi-Head Self-Attention",
        "num_heads": 96,
        "num_layers": 96,
        "context_window": 128000,
        "attention_pattern": f"Each of {n_tokens} tokens attends to all previous tokens",
        "tokens": tokens,
        "attention_matrix_head_0": attention_matrix,
        "explanation": (
            "The transformer calculates attention scores between tokens to understand "
            "relationships and dependencies. This is one of 96 attention heads in one "
            "of 96 layers."
        ),
        "code": (
            "Q = embeddings @ W_query  # Query matrix\n"
            "K = embeddings @ W_key    # Key matrix\n"
            "V = embeddings @ W_value  # Value matrix\n"
            "\n"
            "attention_scores = (Q @ K.T) / sqrt(d_k)\n"
            "attention_weights = softmax(attention_scores)\n"
            "output = attention_weights @ V\n"
            "\n"
            "# This happens 96 times (num_heads) per layer\n"
            "# Across 96 layers in GPT-4"
        ),
    }

    # =========================================================================
    # Layer 8: Next-Token Prediction (real logprobs from API)
    # =========================================================================
    print("Layer 8: Next-Token Prediction (calling OpenAI API)")

    # Build the context with injected knowledge
    context_block = "\n".join(f"- {doc['snippet']}" for doc in top_results)
    system_msg = (
        "You are a helpful assistant. Use the following context about the user:\n"
        f"{context_block}\n\n"
        "Respond naturally and helpfully."
    )

    prediction_resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
        logprobs=True,
        top_logprobs=5,
    )

    choice = prediction_resp.choices[0]
    full_response = choice.message.content

    # Extract logprobs for the first several tokens
    first_token_logprobs = []
    if choice.logprobs and choice.logprobs.content:
        for i, token_lp in enumerate(choice.logprobs.content[:20]):
            entry = {
                "position": i,
                "chosen_token": token_lp.token,
                "logprob": round(token_lp.logprob, 4),
                "probability": round(np.exp(token_lp.logprob), 4),
            }
            if token_lp.top_logprobs:
                entry["alternatives"] = [
                    {
                        "token": alt.token,
                        "logprob": round(alt.logprob, 4),
                        "probability": round(np.exp(alt.logprob), 4),
                    }
                    for alt in token_lp.top_logprobs
                ]
            first_token_logprobs.append(entry)

    layer8 = {
        "vocab_size": encoder.n_vocab,
        "temperature": 0.7,
        "top_p": 0.9,
        "model": CHAT_MODEL,
        "first_token_predictions": first_token_logprobs,
        "code": (
            "logits = transformer_output  # [vocab_size] probabilities\n"
            "probs = softmax(logits / temperature)\n"
            "\n"
            "# Top-p (nucleus) sampling\n"
            "sorted_probs = sort(probs, descending=True)\n"
            "cumsum = cumulative_sum(sorted_probs)\n"
            "top_p_mask = cumsum <= 0.9\n"
            "\n"
            "# Sample from top-p tokens\n"
            "next_token = sample(probs[top_p_mask])"
        ),
    }

    # =========================================================================
    # Layer 9: Detokenization (Decode)
    # =========================================================================
    print("Layer 9: Detokenization")

    response_token_ids = encoder.encode(full_response)
    # Show a "raw" version — lowercase, no markdown
    raw_response = full_response.lower()
    for char in ["*", "#", "-", "\n"]:
        raw_response = raw_response.replace(char, " ")
    # Collapse whitespace
    raw_response = " ".join(raw_response.split())

    layer9 = {
        "token_ids": response_token_ids[:30],
        "total_response_tokens": len(response_token_ids),
        "decoded_text": raw_response[:300]
        + ("..." if len(raw_response) > 300 else ""),
        "decoding_method": f"tiktoken ({ENCODING_NAME})",
        "code": (
            "import tiktoken\n"
            "\n"
            f'decoder = tiktoken.get_encoding("{ENCODING_NAME}")\n'
            f"token_ids = {response_token_ids[:8]}...\n"
            "text = decoder.decode(token_ids)\n"
            f'# Output: "{raw_response[:60]}..."'
        ),
    }

    # =========================================================================
    # Layer 10: Post-Processing & Formatting
    # =========================================================================
    print("Layer 10: Post-Processing & Formatting")

    layer10 = {
        "raw_output": raw_response[:500] + ("..." if len(raw_response) > 500 else ""),
        "formatted_output": full_response,
        "formatting_rules": [
            "Capitalize proper nouns and sentence starts",
            "Add markdown formatting (bold, headers)",
            "Structure with bullet points or numbered lists",
            "Add descriptive section headers",
            "Ensure consistent tense and tone",
        ],
        "code": (
            "def format_response(raw_text):\n"
            "    formatted = capitalize_sentences(raw_text)\n"
            "    formatted = add_markdown_headers(formatted)\n"
            "    formatted = create_bullet_points(formatted)\n"
            "    formatted = fix_spacing(formatted)\n"
            "    formatted = ensure_consistency(formatted)\n"
            "    return formatted"
        ),
    }

    # =========================================================================
    # Assemble full output
    # =========================================================================
    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "models_used": {
                "embeddings": EMBEDDING_MODEL,
                "chat": CHAT_MODEL,
                "tokenizer": ENCODING_NAME,
            },
        },
        "userMessage": prompt,
        "layer1_rawInput": layer1,
        "layer2_tokenization": layer2,
        "layer3_embeddings": layer3,
        "layer4_semanticSearch": layer4,
        "layer5_contextInjection": layer5,
        "layer6_taskDecomposition": layer6,
        "layer7_attention": layer7,
        "layer8_prediction": layer8,
        "layer9_detokenization": layer9,
        "layer10_postProcessing": layer10,
        "finalResponse": full_response,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10-layer LLM architecture data for visualization"
    )
    parser.add_argument("prompt", help="The user prompt to process")
    parser.add_argument(
        "-o", "--output", help="Output file path (default: auto-generated in output/)"
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("  export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)

    result = generate_layer_data(args.prompt)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs("output", exist_ok=True)
        # Create filename from prompt
        slug = args.prompt[:40].lower()
        slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
        slug = slug.strip().replace(" ", "_")
        output_path = f"output/{slug}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone! Output written to: {output_path}")
    print(f"Total tokens in response: {result['layer9_detokenization']['total_response_tokens']}")


if __name__ == "__main__":
    main()
