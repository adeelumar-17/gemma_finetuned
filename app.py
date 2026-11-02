"""
Insurance LLM Demo — Streamlit App
====================================
Inference via HuggingFace Inference API (InferenceClient).
No local model loading — works on Streamlit Cloud free tier.

Three models:
  1. Gemma-4-E4B fine-tuned on Bitext Insurance Chatbot dataset
  2. Gemma-4-E4B fine-tuned on InsuranceQA v2 dataset
  3. Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B  (public baseline)

HF token is read exclusively from st.secrets["HF_TOKEN"] (set in Streamlit
Cloud dashboard → App Settings → Secrets). Never entered in the UI.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import streamlit as st
from huggingface_hub import InferenceClient

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insurance LLM Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# NOTE: for your fine-tuned models the `model_id` must point to a full model
# (or merged adapter) on HF Hub that is supported by the Inference API.
# If your repos only contain LoRA adapter weights, merge + push them first:
#   model.merge_and_unload().push_to_hub("adeelumar17/<repo-name>")
# ──────────────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "Gemma-4-E4B · Bitext (fine-tuned)": {
        "model_id": "adeelumar17/bitext",
        "kind": "bitext",
        "description": (
            "**Gemma 4 E4B** fine-tuned with QLoRA on the "
            "[Bitext Insurance Chatbot](https://huggingface.co/datasets/bitext/"
            "Bitext-insurance-llm-chatbot-training-dataset) dataset "
            "(39 000 synthetic customer-support pairs). Trained by **adeelumar17**."
        ),
        "system_prompt": (
            "You are a helpful insurance customer-support assistant. "
            "Answer the customer's question clearly and concisely."
        ),
    },
    "Gemma-4-E4B · InsuranceQA (fine-tuned)": {
        "model_id": "adeelumar17/insuranceqa_20261031_194809",
        "kind": "insuranceqa",
        "description": (
            "**Gemma 4 E4B** fine-tuned with QLoRA on "
            "[InsuranceQA v2](https://github.com/shuzi/insuranceQA/tree/master/V2) "
            "(≈12 856 real customer questions answered by insurance professionals). "
            "Trained by **adeelumar17**."
        ),
        "system_prompt": (
            "You are a knowledgeable insurance expert. "
            "Provide accurate, professional answers to insurance questions."
        ),
    },
    "Llama-3-8B · Open-Insurance (public baseline)": {
        "model_id": "Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B",
        "kind": "general",
        "description": (
            "**[Open-Insurance-LLM-Llama3-8B]"
            "(https://huggingface.co/Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B)** "
            "— publicly available Llama-3-8B fine-tuned on insurance domain data "
            "by Raj Maharajwala. Used here as a comparison baseline."
        ),
        "system_prompt": (
            "You are an insurance expert assistant. "
            "Answer questions about insurance policies, claims, and coverage clearly."
        ),
    },
}

MODEL_NAMES = list(MODELS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Token — from Streamlit Cloud Secrets or environment variable only
# ──────────────────────────────────────────────────────────────────────────────

def get_hf_token() -> Optional[str]:
    """Read HF token from st.secrets or env var. Never from UI."""
    try:
        token = st.secrets.get("HF_TOKEN", "")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or None


HF_TOKEN = get_hf_token()

# ──────────────────────────────────────────────────────────────────────────────
# Inference via HF Inference API
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client(token: Optional[str]) -> InferenceClient:
    """One shared InferenceClient per session."""
    return InferenceClient(token=token)


def run_inference(
    model_name: str,
    user_message: str,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> tuple[str, float]:
    """Call HF Inference API with chat completions or text_generation fallback."""
    client = get_client(HF_TOKEN)
    model_id = MODELS[model_name]["model_id"]

    # 1. Try chat.completions.create (modern conversational format)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs: dict = dict(
            model=model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            stream=False,
        )
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        else:
            kwargs["temperature"] = 0.01  # near-greedy

        t0 = time.time()
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0

        text = response.choices[0].message.content or ""
        return text.strip(), elapsed

    except Exception as chat_err:
        # 2. Fallback to text_generation endpoint if conversational task isn't configured
        prompt = f"System: {system_prompt}\nUser: {user_message}\nAssistant:"
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.time()
        try:
            raw_text = client.text_generation(prompt, model=model_id, **gen_kwargs)
            elapsed = time.time() - t0
            return raw_text.strip(), elapsed
        except Exception:
            # If both fail, re-raise the original error for clarity
            raise chat_err


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — inference settings only (no token input)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Token status indicator (no input box — set via Streamlit Cloud Secrets)
    if HF_TOKEN:
        st.success("🔑 HF token loaded from secrets", icon="✅")
    else:
        st.error(
            "🔑 No HF token found.\n\n"
            "Add it in **Streamlit Cloud → App Settings → Secrets**:\n"
            "```\nHF_TOKEN = \"hf_...\"\n```",
            icon="🚨",
        )

    st.markdown("---")
    st.subheader("Generation Settings")

    do_sample = st.checkbox("Sampling (vs. near-greedy)", value=True)

    temperature = st.slider(
        "Temperature", 0.1, 2.0, 0.9, 0.05,
        disabled=not do_sample,
        help="Higher = more creative. Lower = more focused.",
    )
    top_p = st.slider(
        "Top-p (nucleus)", 0.1, 1.0, 0.95, 0.01,
        disabled=not do_sample,
    )
    repetition_penalty = st.slider(
        "Repetition Penalty", 1.0, 2.0, 1.1, 0.05,
        help="Penalises repeated tokens. 1.0 = no penalty.",
    )
    max_new_tokens = st.slider(
        "Max New Tokens", 64, 1024, 256, 32,
    )

    st.markdown("---")
    st.caption(
        "Inference runs on **HuggingFace's servers** via the Inference API — "
        "no GPU required on Streamlit Cloud."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────

st.title("🛡️ Insurance LLM Demo")
st.caption(
    "Compare three insurance-domain language models — "
    "two fine-tuned by **adeelumar17** on Gemma 4 E4B, "
    "and one public Llama-3 baseline."
)

tab_single, tab_compare, tab_about = st.tabs(
    ["🤖 Single Model", "⚖️ Side-by-Side Compare", "ℹ️ About"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single Model
# ──────────────────────────────────────────────────────────────────────────────

with tab_single:
    st.subheader("Single Model Inference")

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_model = st.selectbox("Select model", MODEL_NAMES, key="single_model_select")
    with col_info:
        st.info(MODELS[selected_model]["description"])

    sys_prompt_single = st.text_area(
        "System Prompt",
        value=MODELS[selected_model]["system_prompt"],
        height=80,
        key="single_sys_prompt",
        help="Instructions given to the model before your message.",
    )

    example_prompts = [
        "What does home insurance cover in case of a flood?",
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
        elif not HF_TOKEN:
            st.error("HF token is missing. Add it in Streamlit Cloud Secrets.")
        else:
            with st.spinner(f"Calling **{selected_model}** via HF Inference API…"):
                try:
                    output, elapsed = run_inference(
                        model_name=selected_model,
                        user_message=user_input_single,
                        system_prompt=sys_prompt_single,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                    )
                    st.success(f"✅ Generated in **{elapsed:.1f}s**")
                    st.markdown("### Response")
                    st.markdown(output)
                    with st.expander("📋 Raw text"):
                        st.code(output)
                    st.caption(f"≈ {len(output.split())} words | {len(output)} chars")

                except Exception as exc:
                    st.error(f"❌ API error: `{exc}`")
                    st.info(
                        "**Common causes:**\n"
                        "- Model not available on HF Inference API "
                        "(adapter-only repos need to be merged + pushed as full models)\n"
                        "- Rate limit on the free Inference API tier "
                        "(try again in a moment)\n"
                        "- Invalid or expired HF token in Secrets"
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Side-by-Side Comparison
# ──────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Side-by-Side Model Comparison")
    st.caption("Select two models and compare their responses to the same prompt.")

    col_a, col_b = st.columns(2)

    with col_a:
        model_a = st.selectbox("Model A", MODEL_NAMES, index=0, key="cmp_model_a")
        st.info(MODELS[model_a]["description"])
        sys_a = st.text_area(
            "System Prompt A",
            value=MODELS[model_a]["system_prompt"],
            height=80,
            key="cmp_sys_a",
        )

    with col_b:
        model_b = st.selectbox("Model B", MODEL_NAMES, index=2, key="cmp_model_b")
        st.info(MODELS[model_b]["description"])
        sys_b = st.text_area(
            "System Prompt B",
            value=MODELS[model_b]["system_prompt"],
            height=80,
            key="cmp_sys_b",
        )

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
        elif not HF_TOKEN:
            st.error("HF token is missing. Add it in Streamlit Cloud Secrets.")
        else:
            results: dict = {}
            col_out_a, col_out_b = st.columns(2)

            for model_name, sys_p, col in [
                (model_a, sys_a, col_out_a),
                (model_b, sys_b, col_out_b),
            ]:
                with col:
                    st.markdown(f"#### {model_name}")
                    with st.spinner(f"Calling API…"):
                        try:
                            out, elapsed = run_inference(
                                model_name=model_name,
                                user_message=user_input_cmp,
                                system_prompt=sys_p,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                do_sample=do_sample,
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

            # Comparison stats
            outs = {k: v for k, v in results.items() if v[0] is not None}
            if len(outs) == 2:
                with st.expander("📊 Quick comparison stats"):
                    names = list(outs.keys())
                    r0, t0_ = outs[names[0]]
                    r1, t1_ = outs[names[1]]
                    st.markdown(
                        f"| | {names[0]} | {names[1]} |\n"
                        f"|---|---|---|\n"
                        f"| Words | {len(r0.split())} | {len(r1.split())} |\n"
                        f"| Characters | {len(r0)} | {len(r1)} |\n"
                        f"| Time (s) | {t0_:.1f} | {t1_:.1f} |"
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — About
# ──────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.subheader("About This Demo")
    st.markdown("""
This Streamlit application demonstrates three insurance-domain language models,
built as part of a fine-tuning evaluation project.
Inference runs via the **HuggingFace Inference API** — no GPU is needed on the server hosting this app.

---

### Models

| Model | Base | Dataset | Trained by |
|---|---|---|---|
| **Gemma-4-E4B · Bitext** | google/gemma-4-E4B-it | Bitext Insurance Chatbot (39 000 rows) | adeelumar17 |
| **Gemma-4-E4B · InsuranceQA** | google/gemma-4-E4B-it | InsuranceQA v2 (≈12 856 Q&A pairs) | adeelumar17 |
| **Llama-3-8B · Open-Insurance** | meta-llama/Meta-Llama-3-8B | Mixed insurance corpus | Raj-Maharajwala |

---

### Training Details (Gemma models)

Both Gemma models were fine-tuned using:
- **QLoRA** — rank 16, alpha 32, dropout 0.05
- **Unsloth** for fast 4-bit training on Kaggle T4 × 2
- **Learning rate**: 1 × 10⁻⁴
- **Task**: Generate synthetic insurance data records matching the training distribution

---

### Architecture

```
app.py
 ├── Sidebar  ← generation settings + token status
 ├── Tab 1: Single Model
 │     └── model select → system prompt → user message → HF Inference API → response
 ├── Tab 2: Side-by-Side Compare
 │     └── two models → shared prompt → parallel API calls → diff stats
 └── Tab 3: About (this page)
```
""")
