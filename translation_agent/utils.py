import os
from typing import List, Union
import time
import openai
from icecream import ic
import json
import hashlib
from dotenv import load_dotenv
from .prompts import (
    TRANSLATION_SYSTEM_MESSAGE,
    INITIAL_TRANSLATION_PROMPT,
    REFLECTION_SYSTEM_MESSAGE,
    REFLECTION_PROMPT,
    REFLECTION_PROMPT_WITH_COUNTRY,
    IMPROVEMENT_SYSTEM_MESSAGE,
    IMPROVEMENT_PROMPT
)

# 加载 .env 文件
load_dotenv()

# 替换原有的 key 定义
silicon_flow_api_key = os.getenv("SILICON_FLOW_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# 配置 clients
base_url = 'https://api.siliconflow.cn/v1'
client = openai.OpenAI(api_key=silicon_flow_api_key, base_url=base_url)

# 配置 Google API
genai.configure(api_key=google_api_key)

MAX_TOKENS_PER_CHUNK = (
    500  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time

import google.generativeai as genai

CURRENT_MODEL = 'gemini-exp-1206'#'Qwen/Qwen2.5-72B-Instruct'
CURRENT_MODEL = 'gemini-2.0-flash-experimental'
CURRENT_MODEL = 'gemini-1.5-flash'
CURRENT_MODEL = 'gemini-2.0-flash-exp'

CACHE_DIR = "completion_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(prompt: str, system_message: str, model: str) -> str:
    """生成缓存键值"""
    combined = f"{prompt}|{system_message}|{model}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = CURRENT_MODEL,
    temperature: float = 0.3,
    google: bool = False,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    # 生成缓存文件路径
    cache_key = get_cache_key(prompt, system_message, model)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    # 检查缓存是否存在
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)['response']
    
    # 如果缓存不存在，调用原有的完成函数逻辑
    if 'gemini'in model:
        model = genai.GenerativeModel(model)
        response = model.generate_content(system_message + "\t" + prompt,
            generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            temperature=0.5,
            ),
        )
        response_text = response.text
    else:
        if json_mode:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=1,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.choices[0].message.content
    
    # 保存结果到缓存
    cache_data = {
        'prompt': prompt,
        'system_message': system_message,
        'model': model,
        'response': response_text
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    return response_text


def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    system_message = TRANSLATION_SYSTEM_MESSAGE.format(
        source_lang=source_lang,
        target_lang=target_lang
    )

    translation_prompt = INITIAL_TRANSLATION_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text
    )

    translation = get_completion(translation_prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for the target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = REFLECTION_SYSTEM_MESSAGE.format(
        source_lang=source_lang,
        target_lang=target_lang
    )

    if country:
        reflection_prompt = REFLECTION_PROMPT_WITH_COUNTRY.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            translation_1=translation_1,
            country=country
        )
    else:
        reflection_prompt = REFLECTION_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            translation_1=translation_1
        )

    reflection = get_completion(reflection_prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = IMPROVEMENT_SYSTEM_MESSAGE.format(
        source_lang=source_lang,
        target_lang=target_lang
    )

    prompt = IMPROVEMENT_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
        reflection=reflection
    )

    translation_2 = get_completion(prompt, system_message=system_message)

    return translation_2


def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for the target language.
    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return (translation_1, reflection, translation_2)


def split_text(text, max_length):
    lines = text.split('\n')
    splits = []
    segment = ''
    for line in lines:
        segment += line + '\n'
        if len(segment) > max_length:
            splits.append(segment)
            segment = ''
    splits.append(segment)
    return splits


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    print(len(source_text))

    """Translate the source_text from source_lang to target_lang."""
    ic("Translating text as a single chunk")

    final_translation_triple = one_chunk_translate_text(
        source_lang, target_lang, source_text, country
    )

    return final_translation_triple

