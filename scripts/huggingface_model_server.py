import argparse
import fastapi
import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import logging
import json
from agent_studio.utils.communication import bytes2str, str2bytes
from agent_studio.utils.types import MessageList
from typing import Any
import numpy as np
from pathlib import Path
from PIL import Image
import time
from transformers import StoppingCriteria, StoppingCriteriaList


logger = logging.getLogger(__name__)

app = fastapi.FastAPI()

# Load the model and processor once
model_id = "google/gemma-3n-e4b-it"
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_id)


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

@app.post("/generate")
async def generate(request: GenerateRequest) -> JSONResponse:
    messages_decoded = str2bytes(request.messages)
    gemma_input_messages = convert_message_to_gemma_format(messages_decoded)
    # Process the input messages
    inputs = processor.apply_chat_template(
        gemma_input_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    logger.info("Starting model inference!")
    timing_criteria = TimingStoppingCriteria()
    stopping_criteria = StoppingCriteriaList([timing_criteria])
    total_start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=10000, 
            do_sample=False,
            stopping_criteria=stopping_criteria
        )
        generation = generation[0][input_len:]
    total_time = time.time() - total_start_time
    
    # Log timing information
    if timing_criteria.token_times:
        avg_token_time = sum(timing_criteria.token_times) / len(timing_criteria.token_times)
        tokens_per_second = 1 / avg_token_time if avg_token_time > 0 else 0
        logger.info(f"Generated {len(timing_criteria.token_times)} tokens in {total_time:.2f}s")
        # logger.info(f"Average per-token time: {avg_token_time:.4f}s ({tokens_per_second:.2f} tokens/sec)")
    
    logger.info("Model inference complete!")
    decoded = processor.decode(generation, skip_special_tokens=True)
    return JSONResponse(content={
        "message": bytes2str(decoded),
        "info": bytes2str({"timing": {
            "total_time": total_time,
            "token_count": len(timing_criteria.token_times),
            "tokens_per_second": tokens_per_second if 'tokens_per_second' in locals() else 0
        }})
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=64000, help="Port to run the server on"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
