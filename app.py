"""
Insurance LLM Demo — Streamlit App
====================================
Three models:
  1. Gemma-4-E4B fine-tuned on Bitext Insurance Chatbot dataset  (QLoRA adapter)
  2. Gemma-4-E4B fine-tuned on InsuranceQA v2 dataset            (QLoRA adapter)
  3. Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B                (public HF model)

Modes:
  • Single Model  — try one model at a time
  • Side-by-Side  — compare any two models simultaneously
"""

from __future__ import annotations

import gc
import os
import time
from typing import Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODELS = {
    "Gemma-4-E4B · Bitext (fine-tuned)": {
        "base_model_id": "google/gemma-4-E4B-it",
        "adapter_id": "adeelumar17/bitext",          # HF Hub adapter repo
        "kind": "bitext",
        "description": (
            "**Gemma 4 E4B** fine-tuned with QLoRA on the "
            "[Bitext Insurance Chatbot](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) "
            "dataset (39 000 synthetic customer-support pairs). "
            "Trained by **adeelumar17** on Kaggle T4 × 2."
        ),
        "system_prompt": (
            "You are a helpful insurance customer-support assistant. "
            "Answer the customer's question clearly and concisely."
        ),
        "is_lora": True,
        "framework": "unsloth",
    },
    "Gemma-4-E4B · InsuranceQA (fine-tuned)": {
        "base_model_id": "google/gemma-4-E4B-it",
        "adapter_id": "adeelumar17/insuranceqa_20261031_194809",   # HF Hub backup repo
        "kind": "insuranceqa",
        "description": (
            "**Gemma 4 E4B** fine-tuned with QLoRA on "
            "[InsuranceQA v2](https://github.com/shuzi/insuranceQA/tree/master/V2) "
            "(≈12 856 real customer questions answered by insurance professionals). "
            "Trained by **adeelumar17** on Kaggle T4 × 2."
        ),
        "system_prompt": (
            "You are a knowledgeable insurance expert. "
            "Provide accurate, professional answers to insurance questions."
        ),
        "is_lora": True,
        "framework": "unsloth",
    },
    "Llama-3-8B · Open-Insurance (public)": {
        "base_model_id": "Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B",
        "adapter_id": None,
        "kind": "general",
        "description": (
            "**[Open-Insurance-LLM-Llama3-8B](https://huggingface.co/Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B)** "
            "— a publicly available Llama-3-8B model fine-tuned on insurance domain data "
            "by Raj Maharajwala. Included here as a comparison baseline."
        ),
        "system_prompt": (
            "You are an insurance expert assistant. "
            "Answer questions about insurance policies, claims, and coverage clearly."
        ),
        "is_lora": False,
        "framework": "transformers",
    },
}

MODEL_NAMES = list(MODELS.keys())

DEFAULT_SYSTEM_PROMPTS = {k: v["system_prompt"] for k, v in MODELS.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insurance LLM Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — model loading (cached per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str, load_in_4bit: bool, hf_token: Optional[str]):
    """Load and cache a model + tokenizer pair."""
    cfg = MODELS[model_name]
    base_id = cfg["base_model_id"]
    adapter_id = cfg["adapter_id"]
    is_lora = cfg["is_lora"]
    framework = cfg["framework"]

    token_kwarg = {"token": hf_token} if hf_token else {}

    if framework == "unsloth":
        try:
            from unsloth import FastModel  # type: ignore
            import torch
            from peft import PeftModel  # type: ignore

            model, processor = FastModel.from_pretrained(
                model_name=base_id,
                max_seq_length=4096,
                dtype=torch.float16,
                load_in_4bit=load_in_4bit,
                full_finetuning=False,
                **token_kwarg,
            )
            tokenizer = getattr(processor, "tokenizer", processor)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            if is_lora and adapter_id:
                model = PeftModel.from_pretrained(model, adapter_id, **token_kwarg)
                model = model.merge_and_unload()

            model.eval()
            FastModel.for_inference(model)
            return model, tokenizer, "unsloth"

        except ImportError:
            # Fallback: load with standard transformers + bitsandbytes
            framework = "transformers"

    # Standard transformers path (also used as fallback from unsloth)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    bnb_config = None
    if load_in_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(base_id, **token_kwarg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict = dict(
        device_map="auto",
        dtype=torch.float16,
        **token_kwarg,
    )
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)

    if is_lora and adapter_id:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, adapter_id, **token_kwarg)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, "transformers"


def unload_model(model_name: str):
    """Remove a model from the cache and free GPU memory."""
    load_model_cached.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    model_name: str,
    user_message: str,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
    load_in_4bit: bool,
    hf_token: Optional[str],
) -> tuple[str, float]:
    """Return (generated_text, elapsed_seconds)."""
    import torch

    cfg = MODELS[model_name]

    with st.spinner(f"Loading **{model_name}**… (first load may take a minute)"):
        model, tokenizer, fwk = load_model_cached(model_name, load_in_4bit, hf_token)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Try apply_chat_template first; fall back to raw concatenation
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            # Gemma-4 uses thinking mode; disable it for direct answers
            enable_thinking=False if cfg.get("base_model_id", "").startswith("google/gemma-4") else True,
        )
    except Exception:
        raw = f"System: {system_prompt}\nUser: {user_message}\nAssistant:"
        input_ids = tokenizer(raw, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        gen_kwargs["do_sample"] = False

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.time() - t0

    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — global settings
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Auto-load token from .streamlit/secrets.toml or environment variable.
    # Falls back to manual entry so the app still works without a secrets file.
    _default_token = (
        st.secrets.get("HF_TOKEN", "")
        if hasattr(st, "secrets")
        else os.environ.get("HF_TOKEN", "")
    )

    hf_token = st.text_input(
        "🔑 HuggingFace Token",
        value=_default_token,
        type="password",
        help=(
            "Required for Gemma (gated model) and any private adapter repos. "
            "Get yours at huggingface.co/settings/tokens. "
            "You can also store it in .streamlit/secrets.toml as HF_TOKEN = '...' "
            "so it loads automatically."
        ),
    )
    if not hf_token:
        st.warning(
            "⚠️ No HF token found. Gemma models will fail to load. "
            "Add your token above or in `.streamlit/secrets.toml`.",
            icon="🔑",
        )

    st.markdown("---")
    st.subheader("Inference Configuration")

    load_in_4bit = st.checkbox(
        "Load in 4-bit (QLoRA)", value=True,
        help="Reduces VRAM usage significantly. Recommended for GPUs with ≤16 GB VRAM.",
    )

    do_sample = st.checkbox("Sampling (vs. Greedy)", value=True)

    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=2.0, value=0.9, step=0.05,
        disabled=not do_sample,
        help="Higher = more creative/random. Lower = more deterministic.",
    )
    top_p = st.slider(
        "Top-p (nucleus)", min_value=0.1, max_value=1.0, value=0.95, step=0.01,
        disabled=not do_sample,
    )
    top_k = st.slider(
        "Top-k", min_value=0, max_value=200, value=50, step=5,
        disabled=not do_sample,
    )
    repetition_penalty = st.slider(
        "Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05,
        help="Penalises repeated tokens. 1.0 = no penalty.",
    )
    max_new_tokens = st.slider(
        "Max New Tokens", min_value=64, max_value=1024, value=256, step=32,
    )

    st.markdown("---")
    st.caption(
        "**Models**\n"
        "- adeelumar17/bitext *(Gemma-4 + Bitext adapter)*\n"
        "- adeelumar17/insuranceqa_20261031_194809 *(Gemma-4 + InsuranceQA adapter)*\n"
        "- Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B *(public)*"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────

st.title("🛡️ Insurance LLM Demo")
st.caption(
    "Explore and compare three insurance-domain language models — "
    "two fine-tuned by **adeelumar17** on Gemma 4 E4B, "
    "and one public Llama-3 baseline."
)

tab_single, tab_compare, tab_about = st.tabs(["🤖 Single Model", "⚖️ Side-by-Side Compare", "ℹ️ About"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single Model
# ──────────────────────────────────────────────────────────────────────────────

with tab_single:
    st.subheader("Single Model Inference")

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_model = st.selectbox(
            "Select model", MODEL_NAMES, key="single_model_select"
        )
    with col_info:
        st.info(MODELS[selected_model]["description"])

    sys_prompt_single = st.text_area(
        "System Prompt",
        value=DEFAULT_SYSTEM_PROMPTS[selected_model],
        height=80,
        key="single_sys_prompt",
        help="Instructions given to the model before your message.",
    )

    # Example prompts
    example_prompts = [
        "What does my home insurance cover in case of a flood?",
        "How do I file a car insurance claim after an accident?",
        "What is the difference between term life and whole life insurance?",
        "My health insurance claim was denied — what are my options?",
        "Can you explain what a deductible is in simple terms?",
        "What is covered under a standard renters insurance policy?",
    ]
    with st.expander("💡 Example prompts — click to use"):
        for ep in example_prompts:
            if st.button(ep, key=f"single_ex_{ep[:30]}"):
                st.session_state["single_user_input"] = ep
                st.rerun()

    # When key= is used, Streamlit manages the value via session state automatically.
    # Do NOT also pass value=st.session_state.get(key) — that causes a conflict error.
    user_input_single = st.text_area(
        "Your message / question",
        height=120,
        placeholder="Ask anything about insurance…",
        key="single_user_input",
    )

    generate_btn = st.button("🚀 Generate", type="primary", key="single_gen")

    if generate_btn:
        if not user_input_single.strip():
            st.warning("Please enter a question or message.")
        else:
            try:
                output, elapsed = run_inference(
                    model_name=selected_model,
                    user_message=user_input_single,
                    system_prompt=sys_prompt_single,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    load_in_4bit=load_in_4bit,
                    hf_token=hf_token or None,
                )
                st.success(f"✅ Generated in **{elapsed:.1f}s**")
                st.markdown("### Response")
                st.markdown(output)

                with st.expander("📋 Raw text"):
                    st.code(output)

                # Token count estimate
                word_count = len(output.split())
                st.caption(f"≈ {word_count} words | {len(output)} chars")

            except Exception as exc:
                st.error(f"❌ Error during inference:\n```\n{exc}\n```")
                st.info(
                    "Common fixes:\n"
                    "- Enable a **GPU runtime** (Kaggle / Colab / local with CUDA)\n"
                    "- Provide a valid **HuggingFace Token** in the sidebar if repos are private\n"
                    "- Reduce **Max New Tokens** or enable **4-bit quantisation**"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Side-by-Side Comparison
# ──────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Side-by-Side Model Comparison")
    st.caption("Select two models and compare their responses to the same prompt.")

    col_a, col_b = st.columns(2)

    with col_a:
        model_a = st.selectbox(
            "Model A", MODEL_NAMES, index=0, key="cmp_model_a"
        )
        st.info(MODELS[model_a]["description"])
        sys_a = st.text_area(
            "System Prompt A",
            value=DEFAULT_SYSTEM_PROMPTS[model_a],
            height=80,
            key="cmp_sys_a",
        )

    with col_b:
        model_b = st.selectbox(
            "Model B", MODEL_NAMES, index=2, key="cmp_model_b"
        )
        st.info(MODELS[model_b]["description"])
        sys_b = st.text_area(
            "System Prompt B",
            value=DEFAULT_SYSTEM_PROMPTS[model_b],
            height=80,
            key="cmp_sys_b",
        )

    # Shared prompt
    st.markdown("---")
    example_prompts_cmp = [
        "What is covered under a standard home insurance policy?",
        "How do I cancel my insurance policy?",
        "What happens to my insurance if I miss a payment?",
        "Can I insure a pre-existing medical condition?",
    ]
    with st.expander("💡 Example prompts — click to use"):
        for ep in example_prompts_cmp:
            if st.button(ep, key=f"cmp_ex_{ep[:30]}"):
                st.session_state["cmp_user_input"] = ep
                st.rerun()

    # Same rule: key= is set, so do NOT also pass value=st.session_state.get(key)
    user_input_cmp = st.text_area(
        "Shared prompt (sent to both models)",
        height=120,
        placeholder="Ask anything about insurance…",
        key="cmp_user_input",
    )

    cmp_btn = st.button("⚡ Compare Both Models", type="primary", key="cmp_gen")

    if cmp_btn:
        if not user_input_cmp.strip():
            st.warning("Please enter a question or message.")
        elif model_a == model_b:
            st.warning("Please select two *different* models to compare.")
        else:
            results: dict[str, tuple[str, float] | tuple[None, str]] = {}

            col_out_a, col_out_b = st.columns(2)

            for model_name, sys_p, col in [
                (model_a, sys_a, col_out_a),
                (model_b, sys_b, col_out_b),
            ]:
                with col:
                    st.markdown(f"#### {model_name}")
                    try:
                        out, elapsed = run_inference(
                            model_name=model_name,
                            user_message=user_input_cmp,
                            system_prompt=sys_p,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample,
                            load_in_4bit=load_in_4bit,
                            hf_token=hf_token or None,
                        )
                        results[model_name] = (out, elapsed)
                        st.success(f"✅ {elapsed:.1f}s")
                        st.markdown(out)
                        with st.expander("📋 Raw"):
                            st.code(out)
                        st.caption(f"≈ {len(out.split())} words | {len(out)} chars")
                    except Exception as exc:
                        results[model_name] = (None, str(exc))
                        st.error(f"❌ {exc}")

            # Quick diff summary
            if len(results) == 2:
                outs = [v[0] for v in results.values() if v[0] is not None]
                if len(outs) == 2:
                    with st.expander("📊 Quick comparison stats"):
                        names = list(results.keys())
                        r0, r1 = results[names[0]][0], results[names[1]][0]
                        t0, t1 = results[names[0]][1], results[names[1]][1]
                        st.markdown(
                            f"| | {names[0]} | {names[1]} |\n"
                            f"|---|---|---|\n"
                            f"| Words | {len(r0.split())} | {len(r1.split())} |\n"
                            f"| Characters | {len(r0)} | {len(r1)} |\n"
                            f"| Time (s) | {t0:.1f} | {t1:.1f} |"
                        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — About
# ──────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.subheader("About This Demo")

    st.markdown("""
This Streamlit application demonstrates three insurance-domain language models,
built as part of a fine-tuning evaluation project.

---

### Models

| Model | Base | Dataset | Trained by |
|---|---|---|---|
| **Gemma-4-E4B · Bitext** | google/gemma-4-E4B-it | Bitext Insurance Chatbot (39 000 rows) | adeelumar17 |
| **Gemma-4-E4B · InsuranceQA** | google/gemma-4-E4B-it | InsuranceQA v2 (≈12 856 Q&A pairs) | adeelumar17 |
| **Llama-3-8B · Open-Insurance** | meta-llama/Meta-Llama-3-8B | Mixed insurance corpus | Raj-Maharajwala |

---

### Datasets

**Bitext Insurance Chatbot Dataset**
- 39 000 synthetic customer-support request/response pairs
- 27 intents across categories like claims, billing, policies
- License: CDLA-Sharing-1.0
- [HuggingFace link](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset)

**InsuranceQA v2**
- 12 889 real customer questions with answers from insurance professionals
- Grouped, intent-stratified 80/10/10 train/val/test split
- Research-use license
- [GitHub link](https://github.com/shuzi/insuranceQA/tree/master/V2)

---

### Training Details

Both Gemma models were fine-tuned using:
- **QLoRA** (rank = 16, alpha = 32, dropout = 0.05)
- **Unsloth** for fast 4-bit training on Kaggle T4 × 2
- **Learning rate**: 1 × 10⁻⁴ with cosine decay
- **Task**: Generate synthetic insurance data records matching training distribution

---

### How to Run Locally

```bash
pip install streamlit transformers peft bitsandbytes accelerate sentencepiece
# If Unsloth is available:
pip install unsloth unsloth_zoo

streamlit run app.py
```

> **GPU recommended.** CPU inference is extremely slow for these model sizes.
> With a 16 GB GPU and 4-bit quantisation, each Gemma-4 E4B adapter loads in ~1–2 minutes.

---

### Architecture

```
app.py
 ├── Sidebar  ← inference hyperparameters + HF token
 ├── Tab 1: Single Model
 │     └── model select → system prompt → user message → generate
 ├── Tab 2: Side-by-Side Compare
 │     └── two model selects → shared prompt → parallel generate + diff stats
 └── Tab 3: About (this page)
```
""")

