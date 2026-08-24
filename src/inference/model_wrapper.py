"""Deterministic model wrapper used by every stage of the pipeline that
touches the target LLM: trace generation (script 02), exact activation
patching (script 08), and attribution patching (script 09).

All generation is greedy / temperature=0 with a fixed seed, per Section 4.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass
class LayerActivations:
    """Residual-stream activations captured at `resid_pre` for every decoder
    layer, for a single forward pass. Shape per layer: [seq_len, hidden]."""
    hidden_states: list[torch.Tensor]  # len == n_layers, one entry per layer input
    input_ids: torch.Tensor


class ModelWrapper:
    def __init__(self, model_cfg: dict, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = model_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_id"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"],
            torch_dtype=dtype,
            device_map=model_cfg.get("device_map", "auto"),
        )
        self.model.eval()
        self.n_layers = model_cfg["n_layers"]
        self.gen_cfg = model_cfg.get("generation", {})
        self._decoder_layers = self._locate_decoder_layers()

    # ------------------------------------------------------------------
    def _locate_decoder_layers(self):
        """Works for Llama/Qwen/Phi3-style HF models (model.model.layers)."""
        base = getattr(self.model, "model", self.model)
        return base.layers

    def _seed_everything(self):
        seed = self.gen_cfg.get("seed", 0)
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    def render_and_tokenize(self, prompt: str):
        if self.cfg.get("chat_template", True):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            # apply_chat_template returns a BatchEncoding (not a bare tensor)
            # on current transformers versions - unwrap it the same way the
            # non-chat-template branch below already does.
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.model.device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self._seed_everything()
        input_ids = self.render_and_tokenize(prompt)
        out = self.model.generate(
            input_ids,
            do_sample=self.gen_cfg.get("do_sample", False),
            num_beams=self.gen_cfg.get("num_beams", 1),
            max_new_tokens=max_new_tokens or self.gen_cfg.get("max_new_tokens", 512),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture_activations(self, full_text: str) -> LayerActivations:
        """Runs one forward pass over `full_text` (prompt + continuation,
        already concatenated) and captures resid_pre for every layer."""
        input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.model.device)
        captured = [None] * self.n_layers
        handles = []

        def make_hook(idx):
            def hook(module, inputs):
                captured[idx] = inputs[0].detach().clone()
            return hook

        for i, layer in enumerate(self._decoder_layers):
            handles.append(layer.register_forward_pre_hook(make_hook(i)))
        try:
            self.model(input_ids)
        finally:
            for h in handles:
                h.remove()
        return LayerActivations(hidden_states=[c[0] for c in captured], input_ids=input_ids[0])

    @contextlib.contextmanager
    def patch_layer(self, layer_idx: int, positions: list[int], replacement: torch.Tensor):
        """Context manager: during the forward pass, overwrite resid_pre of
        `layer_idx` at `positions` with `replacement` (shape [len(positions), hidden])."""
        layer = self._decoder_layers[layer_idx]

        def hook(module, inputs):
            hs = inputs[0].clone()
            for k, pos in enumerate(positions):
                if pos < hs.shape[1]:
                    hs[0, pos, :] = replacement[k].to(hs.dtype).to(hs.device)
            return (hs,) + inputs[1:]

        handle = layer.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def next_token_logits(self, full_text: str, patch: dict | None = None) -> torch.Tensor:
        """Logits for the token following `full_text`. `patch`, if given, is
        {"layer_idx": int, "positions": [...], "replacement": Tensor}."""
        input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.model.device)
        if patch is None:
            out = self.model(input_ids)
        else:
            with self.patch_layer(patch["layer_idx"], patch["positions"], patch["replacement"]):
                out = self.model(input_ids)
        return out.logits[0, -1, :]

    @torch.no_grad()
    def sequence_logprob(self, prefix: str, continuation: str, patch: dict | None = None) -> float:
        """Length-normalized (mean per-token) log-probability of `continuation`
        given `prefix`, optionally with one layer patched. Used by
        src.mechanisms.scoring as the m_i margin ingredient (Section 21)."""
        prefix_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.model.device)
        full = self.tokenizer(prefix + continuation, return_tensors="pt").input_ids.to(self.model.device)
        cont_len = full.shape[1] - prefix_ids.shape[1]
        if cont_len <= 0:
            return float("nan")

        if patch is None:
            out = self.model(full)
        else:
            with self.patch_layer(patch["layer_idx"], patch["positions"], patch["replacement"]):
                out = self.model(full)

        logits = out.logits[0, prefix_ids.shape[1] - 1: full.shape[1] - 1, :]
        targets = full[0, prefix_ids.shape[1]: full.shape[1]]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lps = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return token_lps.mean().item()
