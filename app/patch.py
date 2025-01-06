import time
from functools import wraps
from threading import Lock
from typing import Optional

from loguru import logger
import os
import sys
import openai
import translation_agent.utils as utils
import google.generativeai as genai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RPM = 60
MODEL = ""
TEMPERATURE = 0.3
# Hide js_mode in UI now, update in plan.
CLIENT = ''
JS_MODE = False

# Add your LLMs here
def model_load(
    endpoint: str,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = TEMPERATURE,
    rpm: int = RPM,
    js_mode: bool = JS_MODE,
):
    global CLIENT, RPM, MODEL, TEMPERATURE, JS_MODE, ENDPOINT
    ENDPOINT = endpoint
    RPM = rpm
    MODEL = model
    TEMPERATURE = temperature
    JS_MODE = js_mode
    if endpoint=='Deepseek':
        base_url = 'https://api.deepseek.com'
    logger.info(f'load {model} for {endpoint} from {base_url} api_key {api_key}')
    if endpoint == 'Gemini':
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model)
    elif endpoint == 'OpenAI':
        CLIENT = openai.OpenAI(api_key=api_key)
    else:
        CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url) 


def rate_limit(get_max_per_minute):
    def decorator(func):
        lock = Lock()
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                max_per_minute = get_max_per_minute()
                min_interval = 60.0 / max_per_minute
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed

                if left_to_wait > 0:
                    time.sleep(left_to_wait)

                ret = func(*args, **kwargs)
                last_called[0] = time.time()
                return ret

        return wrapper

    return decorator


one_chunk_initial_translation = utils.one_chunk_initial_translation
one_chunk_reflect_on_translation = utils.one_chunk_reflect_on_translation
one_chunk_improve_translation = utils.one_chunk_improve_translation
one_chunk_translate_text = utils.one_chunk_translate_text
