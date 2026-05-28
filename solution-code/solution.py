"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash
    with the same prompt and return a structured comparison dictionary.
    """
    def _cost(model_id: str, usage: dict) -> float:
        rates = PRICING_1M_TOKENS[model_id]
        return (
            usage["input_tokens"] * rates["input"]
            + usage["output_tokens"] * rates["output"]
        ) / 1_000_000

    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": _cost(OPENAI_MODEL, gpt4o_usage),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": _cost(OPENAI_MINI_MODEL, mini_usage),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": _cost(GEMINI_MODEL, gemini_usage),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    Keeps the last 3 turns (6 messages) of conversation history.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[types.Content] = []

    print("Gemini 2.5 Flash chatbot — type 'quit' or 'exit' to end.")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        print("Bot: ", end="", flush=True)
        assistant_chunks: list[str] = []
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
        )
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                assistant_chunks.append(chunk.text)
        print()

        history.append(
            types.Content(
                role="model",
                parts=[types.Part(text="".join(assistant_chunks))],
            )
        )

        # Keep only the last 3 turns (6 messages).
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list and attach the original prompt.
    """
    results = []
    for prompt in prompts:
        entry = compare_models(prompt)
        entry["prompt"] = prompt
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table.
    """
    model_labels = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    ]

    header = "| Prompt | Model | Response | Latency | Tokens (In/Out) | Cost (USD) |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]

    for entry in results:
        prompt = entry.get("prompt", "")
        prompt_display = (prompt[:47] + "...") if len(prompt) > 50 else prompt
        for key, label in model_labels:
            stats = entry[key]
            response = stats["response"].replace("\n", " ").replace("|", "\\|")
            response_display = (
                (response[:47] + "...") if len(response) > 50 else response
            )
            lines.append(
                f"| {prompt_display} | {label} | {response_display} "
                f"| {stats['latency']:.2f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.6f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
