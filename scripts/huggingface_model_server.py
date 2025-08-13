import argparse
import logging
import time
from pathlib import Path
from typing import Any

import fastapi
import numpy as np
import torch
from peft import PeftConfig, PeftModel
from fastapi import Depends
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    StoppingCriteria,
    BitsAndBytesConfig,
    StoppingCriteriaList,
)

from agent_studio.utils.communication import bytes2str, str2bytes
from agent_studio.utils.types import MessageList

logger = logging.getLogger(__name__)

app = fastapi.FastAPI()

# Global variables for model and processor
model = None
processor = None
model_ready = False
model_name = None


def load_gemma_model(model_id="google/gemma-3n-e4b-it"):
    """Load the Gemma model and processor"""
    global model, processor, model_ready
    logger.info(f"Loading Gemma model: {model_id}")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    model_ready = True  # Set the readiness flag
    return model, processor


def load_qwen_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct", model_weights_path=None):
    """Load the Qwen model and processor"""
    global model, processor, model_ready
    if model_weights_path is None:
        logger.info(f"Loading Qwen model: {model_id}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        model.eval()
    else:
        logger.info(f"Loading local Qwen model weights from: {model_weights_path}")
        # 4-bit quantization config for efficient inference.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model_for_peft = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base_model_for_peft,
            model_weights_path,
            device_map="auto",
            safe_serialization=True,
        )
        model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    model_ready = True
    return model, processor


@app.get("/ready")
async def ready() -> JSONResponse:
    """Endpoint to check if the model is ready"""
    if model_ready:
        return JSONResponse(content={"status": "ready"}, status_code=200)
    else:
        return JSONResponse(content={"status": "loading"}, status_code=503)


class GenerateRequest(BaseModel):
    messages: str

def convert_message_to_gemma_format(
        raw_messages: MessageList,
    ) -> list[dict[str, Any]]:
        """
        Composes the messages to be sent to the model.
        """
        model_message: list[dict[str, Any]] = []
        past_role = None
        for msg in raw_messages:
            if isinstance(msg.content, np.ndarray):
                content = {"type": "image", "image": Image.fromarray(msg.content).convert("RGB")}
            elif isinstance(msg.content, Path):
                content = {"type": "image", "image": Image.open(msg.content).convert("RGB")}
            elif isinstance(msg.content, str):
                content = {"type": "text", "text": msg.content}
            current_role = msg.role
            if past_role != current_role:
                model_message.append(
                    {
                        "role": current_role,
                        "content": [content],
                    }
                )
                past_role = current_role
            else:
                model_message[-1]["content"].append(content)
        return model_message


def convert_message_to_gemma_format(
    raw_messages: MessageList,
) -> list[dict[str, Any]]:
    """
    Composes the messages to be sent to the Gemma model.
    """
    model_message: list[dict[str, Any]] = []
    past_role = None
    for msg in raw_messages:
        if isinstance(msg.content, np.ndarray):
            content = {
                "type": "image",
                "image": Image.fromarray(msg.content).convert("RGB"),
            }
        elif isinstance(msg.content, Path):
            content = {"type": "image", "image": Image.open(msg.content).convert("RGB")}
        elif isinstance(msg.content, str):
            content = {"type": "text", "text": msg.content}
        current_role = msg.role
        if past_role != current_role:
            model_message.append(
                {
                    "role": current_role,
                    "content": [content],
                }
            )
            past_role = current_role
        else:
            model_message[-1]["content"].append(content)
    return model_message


def save_debug_image(img, index=0):
    """Save an image to a file for debugging purposes"""
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    filename = debug_dir / f"debug_image_{timestamp}_{index}.png"
    img.save(filename)
    logger.info(f"Saved debug image to {filename}")
    return filename


def convert_message_to_qwen_format(
    raw_messages: MessageList,
) -> list[dict[str, Any]]:
    """
    Composes the messages to be sent to the Qwen model.
    """
    model_message: list[dict[str, Any]] = []
    past_role = None
    for msg in raw_messages:
        if isinstance(msg.content, np.ndarray):
            # By default the images are bgr, so we need to swap
            # the channel order.
            img = msg.content[:, :, ::-1]
            img = Image.fromarray(img).convert("RGB")
            save_debug_image(img, 0)
            content = {
                "type": "image",
                "image": img,
            }
        elif isinstance(msg.content, Path):
            img = Image.open(msg.content).convert("RGB")
            save_debug_image(img, 0)
            content = {"type": "image", "image": img}
        elif isinstance(msg.content, str):
            content = {"type": "text", "text": msg.content}
        current_role = msg.role
        if past_role != current_role:
            model_message.append(
                {
                    "role": current_role,
                    "content": [content],
                }
            )
            past_role = current_role
        else:
            model_message[-1]["content"].append(content)
    return model_message


class TimingStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.token_times = []
        self.last_time = None
        self.start_time = time.time()
    def __call__(self, input_ids, scores, **kwargs):
        current_time = time.time()
        if self.last_time is not None:
            self.token_times.append(current_time - self.last_time)
        self.last_time = current_time
        return False


def get_model_name() -> str:
    return model_name


@app.post("/generate")
async def generate(
    request: GenerateRequest, model_name: str = Depends(get_model_name)
) -> JSONResponse:
    global model, processor

    messages_decoded = str2bytes(request.messages)

    if "gemma" in model_name:
        # Process for Gemma model
        gemma_input_messages = convert_message_to_gemma_format(messages_decoded)
        inputs = processor.apply_chat_template(
            gemma_input_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        logger.info("Starting Gemma model inference!")
        timing_criteria = TimingStoppingCriteria()
        stopping_criteria = StoppingCriteriaList([timing_criteria])
        total_start_time = time.time()
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=10000,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
            generation = generation[0][input_len:]
        total_time = time.time() - total_start_time
        # Log timing information
        if timing_criteria.token_times:
            avg_token_time = sum(timing_criteria.token_times) / len(
                timing_criteria.token_times
            )
            tokens_per_second = 1 / avg_token_time if avg_token_time > 0 else 0
            logger.info(
                f"Generated {len(timing_criteria.token_times)} "
                f"tokens in {total_time:.2f}s"
            )
        logger.info("Model inference complete!")
        decoded = processor.decode(generation, skip_special_tokens=True)

    elif "Qwen" in model_name:
        # Process for Qwen model
        qwen_input_messages = convert_message_to_qwen_format(messages_decoded)
        # Prepare inputs for Qwen model
        text = processor.apply_chat_template(
            qwen_input_messages, tokenize=False, add_generation_prompt=True
        )
        # DEBUGGING!
        logger.info(text)
        image_inputs, video_inputs = process_vision_info(qwen_input_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        logger.info("Starting Qwen model inference!")
        total_start_time = time.time()
        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=1000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        total_time = time.time() - total_start_time
        logger.info(f"Model inference complete {total_time}!")
        decoded = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    else:
        return JSONResponse(
            content={"error": f"Unknown model type: {model_name}"}, status_code=500
        )

    return JSONResponse(
        content={
            "message": bytes2str(decoded),
            "info": bytes2str(
                {
                    "timing": {
                        "total_time": total_time,
                        "token_count": (
                            len(timing_criteria.token_times) if "gemma" in model_name else 0
                        ),
                        "tokens_per_second": (
                            tokens_per_second if "gemma" in model_name else 0
                        ),
                    }
                }
            ),
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=64000, help="Port to run the server on"
    )
    parser.add_argument(
        "--model", type=str, default="gemma-3n-e4b-it", help="Model id to use"
    )
    parser.add_argument(
        "--model_weights_path", type=str, default=None, help="Path to model weights for a finetuned model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    # Load the selected model
    model_name = args.model

    if "gemma" in model_name:
        load_gemma_model(model_name)
    elif "Qwen" in model_name:
        load_qwen_model(model_name, args.model_weights_path)
    else:
        raise ValueError(
            f"Unknown model type: {model_name}. "
            "Currently only Gemma and Qwen models are supported."
        )

    uvicorn.run(app, host=args.host, port=args.port)
