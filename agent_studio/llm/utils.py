import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        # Matches anything between ```<optional label>\n and \n```
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(
        pattern, response, re.DOTALL
    )  # re.DOTALL makes . match also newlines
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = ""

    return extracted_string


def structured_json_extract_from_response(response: str) -> dict[str, str]:
    # Look for JSON block with proper start and end markers
    start_marker = "```json\n"

    if start_marker in response:
        start_idx = response.find(start_marker) + len(start_marker)
        content = response[start_idx:]
        brace_count = 0
        json_end_idx = 0

        for i, char in enumerate(content):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end_idx = i + 1
                    break

        if json_end_idx > 0:
            extracted_string = content[:json_end_idx]
            # Fix nested code blocks by escaping newlines within them
            nested_code_pattern = r"```[a-zA-Z]*\n(.*?)\n```"

            def escape_newlines(match):
                lang = (
                    match.group(0).split("```")[1].split("\n")[0]
                )  # Extract language identifier
                code_content = match.group(1)
                escaped_content = code_content.replace("\n", "\\n")
                return f"```{lang}\\n{escaped_content}```"

            fixed_string = re.sub(
                nested_code_pattern, escape_newlines, extracted_string, flags=re.DOTALL
            )
            # Replace Python booleans with JSON booleans
            fixed_string = fixed_string.replace(" True", " true").replace(
                " False", " false"
            )
            fixed_string = fixed_string.replace(":True", ":true").replace(
                ":False", ":false"
            )
            try:
                return json.loads(fixed_string)
            except json.JSONDecodeError:
                return {}

    return {}


def openai_encode_image(image: Path | Image.Image | np.ndarray) -> str:
    if isinstance(image, Path):
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_type = image.as_posix().split(".")[-1].lower()
        encoded_image = f"data:image/{image_type};base64,{encoded_image}"
    elif isinstance(image, Image.Image):  # PIL image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_image = f"data:image/jpeg;base64,{encoded_image}"
    elif isinstance(image, np.ndarray):  # cv2 image array
        image = Image.fromarray(image).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_image = f"data:image/jpeg;base64,{encoded_image}"
    else:
        raise ValueError(
            "Invalid image type. Please provide a valid image path, PIL "
            "image, or cv2 image array."
        )

    return encoded_image


def anthropic_encode_image(image: Path | Image.Image | np.ndarray) -> str:
    if isinstance(image, Path):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        raise ValueError(
            "Invalid image type. Please provide a valid image path, PIL "
            "image, or cv2 image array."
        )
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_image


def decode_image(encoded_image: str) -> Image.Image:
    if encoded_image.startswith("data:image"):
        encoded_image = encoded_image.split(",")[-1]
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    return image
