import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _balance_json_block(text: str, start_idx: int) -> str:
    """Return smallest balanced {...} substring starting at start_idx,
    ignoring braces inside JSON strings."""
    i = start_idx
    depth = 0
    in_string = False
    escape = False
    started = False

    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
                if started and depth == 0:
                    return text[start_idx:i+1]
        i += 1
    return text[start_idx:i]


def _escape_quotes_in_backticked_blocks(s: str) -> str:
    pattern = r"```([a-zA-Z0-9_+\-]*)\n(.*?)\n```"
    def repl(m):
        lang, content = m.group(1), m.group(2)
        # escape quotes
        content = content.replace('"', '\\"')
        # escape newlines
        content = content.replace("\n", "\\n")
        return f"```{lang}\\n{content}\\n```"
    return re.sub(pattern, repl, s, flags=re.DOTALL)


def _normalize_json_literals_outside_strings(s: str) -> str:
    """Convert True/False/None to true/false/null only when *not* inside a JSON string."""
    out = []
    i = 0
    in_string = False
    escape = False

    def word_at(idx, word):
        end = idx + len(word)
        if end > len(s) or s[idx:end] != word:
            return False
        before_ok = idx == 0 or (not s[idx-1].isalnum() and s[idx-1] != '_')
        after_ok  = end == len(s) or (not s[end].isalnum() and s[end] != '_')
        return before_ok and after_ok

    while i < len(s):
        ch = s[i]
        out.append(ch)
        if in_string:
            if escape: escape = False
            elif ch == '\\': escape = True
            elif ch == '"': in_string = False
            i += 1; continue
        if ch == '"':
            in_string = True
            i += 1; continue

        replaced = False
        for py, js in (("True","true"),("False","false"),("None","null")):
            if word_at(i, py):
                out.pop()
                out.append(js)
                i += len(py)
                replaced = True
                break
        if not replaced:
            i += 1
    return "".join(out)


def structured_json_extract_from_response(response: str) -> dict:
    """Extract JSON from ```json fenced block or first balanced {...}, while preserving code blocks."""
    fence = "```json\n"
    if fence in response:
        start = response.find(fence) + len(fence)
        candidate = _balance_json_block(response, start)
    else:
        # fallback: first balanced {...}
        first_brace = response.find('{')
        if first_brace == -1:
            return {}
        candidate = _balance_json_block(response, first_brace)

    # 1) Protect code blocks by escaping inner double quotes
    candidate_patched = _escape_quotes_in_backticked_blocks(candidate)

    # 2) Try parse as-is
    try:
        return json.loads(candidate_patched)
    except json.JSONDecodeError:
        pass

    # 3) If needed, normalize Python-ish literals *outside* strings and retry
    normalized = _normalize_json_literals_outside_strings(candidate_patched)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return {}


def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
        flags = re.DOTALL
    elif backtick == "`":
        pattern = r"`(.*?)`"
        flags = 0
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    m = re.search(pattern, response, flags)
    return m.group(1) if m else ""


def parse_strategies(strategies_txt: str) -> list[str]:
    """Given an input text with a bunch of strategies, for solving a task, 
    parse out the strategies individually into a list"""
    # Find all strategy sections using regex
    strategy_pattern = r"(?:\*\*Strategy\s*\d+:|Strategy:)\s*([\s\S]*?)(?=(?:\*\*Strategy\s*\d+:|Strategy:)|---|$)"
    strategies = re.findall(strategy_pattern, strategies_txt)
    # Clean up each strategy text
    result = []
    for i, strategy in enumerate(strategies):
        clean_strategy = strategy.strip()
        result.append(clean_strategy)
    return result


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
