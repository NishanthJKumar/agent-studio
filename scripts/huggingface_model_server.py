import argparse

import fastapi
import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

app = fastapi.FastAPI()

# Load the model and processor once
model_id = "google/gemma-3n-e4b-it"
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_id)


class GenerateRequest(BaseModel):
    messages: str


@app.post("/generate")
async def generate(request: GenerateRequest) -> JSONResponse:
    messages = request.messages

    # Process the input messages
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=10000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return JSONResponse(content={"response": decoded})


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
