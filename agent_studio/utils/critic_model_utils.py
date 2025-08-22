import os
from pathlib import Path
from PIL import Image
import logging
import torch
from qwen_vl_utils import process_vision_info

ANSWER_TAG = "<|answer|>"
LABELS = ["Success.", "Failure."]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def to_dev_dtype(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
        if x.dtype.is_floating_point and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
    return x


def prepare_inputs_from_messages(messages, processor):
    """
    Prepare model inputs from pre-formatted messages for critic model inference.

    Args:
        messages: List of message dictionaries in the format expected by the processor
        processor: The model processor

    Returns:
        dict: Contains 'inputs' ready for model inference
    """
    # Prepare text and inputs for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Check if any message contains images
    has_images = any(
        isinstance(msg.get("content"), list) and
        any(item.get("type") == "image" for item in msg["content"])
        for msg in messages
    )

    if has_images:
        # Process vision info if images are present
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

    return {
        'messages': messages,
        'text': text,
        'inputs': inputs
    }


def predict_from_messages(messages, model, processor):
    """
    Predict Success/Failure from pre-formatted messages using the critic model.

    Args:
        messages: List of message dictionaries in the format expected by the processor
        model: The critic model
        processor: The model processor

    Returns:
        tuple: (predicted_label, scores_dict)
    """
    device = model.device

    # Prepare inputs from messages
    prepared = prepare_inputs_from_messages(messages, processor)
    inputs_no_answer = prepared["inputs"]

    # Move base prompt tensors to device
    input_ids = inputs_no_answer["input_ids"].to(device)
    attn = inputs_no_answer.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Move VLM extras (image/video tensors) to device and correct dtype
    extras = {}
    for k, v in inputs_no_answer.items():
        if k in ("input_ids", "attention_mask", "labels"):
            continue
        extras[k] = to_dev_dtype(v, device)

    # Cache tokenized versions of common tokens
    tag_ids = processor.tokenizer(f"{ANSWER_TAG} ", add_special_tokens=False).input_ids
    label_ids = {y: processor.tokenizer(y, add_special_tokens=False).input_ids for y in LABELS}

    # Score each possible label
    scores = {}
    for label_text in LABELS:
        # Append: ANSWER_TAG + ' ' + label
        lab_ids = label_ids[label_text]
        tail = torch.tensor([tag_ids + lab_ids], device=device)
        new_ids = torch.cat([input_ids, tail], dim=1)
        new_attn = torch.cat([attn, torch.ones_like(tail)], dim=1)

        # Supervise only the label tokens (exclude the tag)
        labels = torch.full_like(new_ids, -100)
        lab_len = len(lab_ids)
        labels[:, -lab_len:] = torch.tensor([lab_ids], device=device)

        # Get model output and compute loss
        with torch.no_grad():
            out = model(input_ids=new_ids, attention_mask=new_attn, labels=labels, **extras)
            scores[label_text] = -out.loss.item() * lab_len  # sum logprob (higher is better)

    # Return the label with highest score
    pred = max(scores, key=scores.get)
    return pred, scores
