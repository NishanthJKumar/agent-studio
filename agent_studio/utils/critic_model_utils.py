import os, hashlib

import logging
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

ANSWER_TAG = "<|answer|>"
LABELS = ["Success.", "Failure."]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]

def _debug_dump_once(inputs_on_device, model, processor, texts=None, tag=""):
    # Print dtypes/devices
    try:
        p = next(model.parameters())
        print(f"[CRITIC DEBUG {tag}] model dtype={p.dtype}, device={p.device}, eval={not model.training}")
    except StopIteration:
        pass
    import torch
    print(f"[CRITIC DEBUG {tag}] TF32 matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn.tf32={torch.backends.cudnn.allow_tf32}")

    # Fingerprint inputs
    for k,v in inputs_on_device.items():
        if isinstance(v, torch.Tensor):
            print(f"[CRITIC DEBUG {tag}] {k}: shape={tuple(v.shape)}, dtype={v.dtype}, hash={_sha(v)}")

    # Two forwards back-to-back on IDENTICAL tensors
    with torch.no_grad():
        out1 = model(**inputs_on_device)
        out2 = model(**inputs_on_device)
    max_abs = (out1.logits - out2.logits).abs().max().item()
    print(f"[CRITIC DEBUG {tag}] max_abs_diff_back_to_back_logits={max_abs:.3e}")


def _make_batched_inputs_for_labels(messages, processor, labels):
    """
    Build a proper batched input by asking the processor to batch
    two copies (one per label) of the SAME multimodal prompt.
    """
    base_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    texts = [f"{base_text}{ANSWER_TAG} {lab}" for lab in labels]

    # Detect and gather vision inputs once from messages
    has_images = any(
        isinstance(msg.get("content"), list) and
        any(item.get("type") == "image" for item in msg["content"])
        for msg in messages
    )

    if has_images:
        image_inputs, video_inputs = process_vision_info(messages)
        # Duplicate the *Python* structures to match the number of texts
        images_batched = [image_inputs for _ in texts]
        videos_batched = [video_inputs for _ in texts] if video_inputs is not None else None
        inputs = processor(
            text=texts,
            images=images_batched,
            videos=videos_batched,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(text=texts, padding=True, return_tensors="pt")

    return inputs


def predict_from_messages(messages, model, processor):
    """
    Single forward pass that scores both labels using processor-batched inputs.
    Returns: (predicted_label: str, scores_dict: {label: float})
    """
    device = model.device

    # 1) Build a *batched* pack with two rows (Success/Failure)
    inputs = _make_batched_inputs_for_labels(messages, processor, LABELS)

    # 2) Move everything to the model device (keep dtypes as returned by processor)
    inputs_on_device = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    # _debug_dump_once(inputs_on_device, model, processor, tag="pre")

    # 3) Forward once
    with torch.no_grad():
        out = model(**inputs_on_device)
    logits = out.logits                                # [B=2, T, V]
    attn   = inputs_on_device["attention_mask"]        # [B=2, T]

    # 4) Compute per-row label logprobs using attention_mask to find the end
    scores = {}
    for i, label in enumerate(LABELS):
        lab_ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        L = len(lab_ids)
        end = int(attn[i].sum().item()) - 1  # last index
        step_logits = logits[i, end - L : end, :].float()  # <-- force fp32 here
        tgt = torch.tensor(lab_ids, device=logits.device, dtype=torch.long).unsqueeze(-1)
        lp = torch.log_softmax(step_logits, dim=-1).gather(-1, tgt).squeeze(-1)
        scores[label] = lp.sum().item()

    pred = max(scores, key=scores.get)
    return pred, scores
