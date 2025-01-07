import os
import re
import time
import re
from tqdm import tqdm
import random
from loguru import logger
import concurrent.futures
import json

import argparse

import time
import hashlib
import openai
from loguru import logger
import google.generativeai as genai
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
siliconflow_api_key = os.getenv("SILICON_FLOW_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# 配置 clients
siliconflow_base_url = 'https://api.siliconflow.cn/v1'
deepseek_base_url='https://api.deepseek.com'

# 配置 Google API
genai.configure(api_key=google_api_key)

MAX_TOKENS_PER_CHUNK = 500  # 每个 chunk 的最大 token 数

CACHE_DIR = "completion_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(prompt: str, system_message: str, model: str) -> str:
    """生成缓存键值"""
    combined = f"{prompt}|{system_message}|{model}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = 'deepseek-chat',
    temperature: float = 1.3,
    client = ''):
    """
    Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model_name (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gemini-2.0-flash-exp".
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
            logger.info(f'{prompt} cached')
            return json.load(f)['response']
    
    # 如果缓存不存在，调用原有的完成函数逻辑
    if 'gemini' in model:
        gemini_model = genai.GenerativeModel(model)
        try:
            response = gemini_model.generate_content(system_message + "\t" + prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                ),
            )
            response_text = response.text
        except Exception as e:
            logger.info(f'Call gemini llm {prompt} throw an exception: {e}')
            return None
    else:
        if not client:
            (key, url) = (siliconflow_api_key, siliconflow_base_url) if 'siliconflow' in model else (deepseek_api_key, deepseek_base_url)
            client = openai.OpenAI(api_key=key, base_url=url)
        try:
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
        except Exception as e:
            logger.info(f'Call openai llm {prompt} throw an exception: {e}')
            return None
    
    # 保存结果到缓存
    cache_data = {
        'prompt': prompt,
        'system_message': system_message,
        'model': model,
        'response': response_text
    }
    logger.info(f'cache_data: {cache_data}')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    return response_text


global_result = dict()
def translate_text(text, index, source_lang, target_lang, country, client=''):
    time.sleep(random.randint(2, 10))
    logger.info(f'translate {text}')
    translation_triple = translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=text,
        country=country,
        client=client
    )
    if translation_triple is not None:
        global_result[index] = translation_triple
    return translation_triple


def translate_multiple_thread(text_splits, source_lang, target_lang, country, max_workers, client=''):
    remained_splits = []
    init_count = len(global_result)
    for i, text in enumerate(text_splits):
        if i not in global_result:
            remained_splits.append([i, text])
    with tqdm(total=len(remained_splits), desc="Translate text") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_translation = {executor.submit(translate_text, pair[1], pair[0], source_lang, target_lang, country, client=client): pair for pair in remained_splits}
            # 提交每个问题的处理
            for future in concurrent.futures.as_completed(future_to_translation):
                (index, text) = future_to_translation[future]
                if not future.result():
                    pbar.update(1)
    logger.info(f'Multi-thread translation done, total: {len(text_splits)}, init: {init_count}, final: {len(text_splits)}')
    return len(global_result) == len(text_splits)


def split_novel(source_text, split_tokens=300):
    splits = re.split(r'(\n\n第.章)', source_text.replace(u'\u3000', ' '))
    chapters = []
    current = ''
    for split in splits:
        if re.match(r'\n\n第.章', split):
            chapters += split_text(current, split_tokens)
            current = split
        else:
            current = current + split
    chapters += split_text(current, split_tokens)
    return chapters


def translate_whole_text(source_text, source_lang, target_lang, country, workers, client=''):
    chapters = split_novel(source_text)
    tries = 0
    MAX_TRIES = 5
    total_time = 0
    while True:
        logger.info(f'Try {tries} time')
        start = time.time()
        status = translate_multiple_thread(chapters, source_lang, target_lang, country, workers, client=client)
        end = time.time()
        total_time += end-start
        logger.info(f'Try {tries} translation done, cost {end-start} seconds. status:{status}')
        if status:
            break
        tries += 1
        if tries >= MAX_TRIES:
            logger.info('exceeed MAX_TRIES {MAX_TRIES}, break')
            break
    status = len(global_result) == len(chapters)
    logger.info(f'Final translation done, cost {total_time} seconds, status:{status}, input:{len(chapters)}, output:{len(global_result)}')
    final_result = []
    for i in range(len(chapters)):
        if i in global_result:
            final_result.append({'source':chapters[i], 'status': 'success'} | global_result[i])
        else:
            final_result.append({'source':chapters[i], 'status': 'failed'})
    return final_result


def one_chunk_initial_translation(
    source_lang, target_lang, source_text, client =''
):
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

    return get_completion(translation_prompt, system_message=system_message, client=client)


def one_chunk_reflect_on_translation(
    source_lang,
    target_lang,
    source_text,
    translation_1,
    country = "", client = ''
):
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

    reflection_prompt = REFLECTION_PROMPT_WITH_COUNTRY.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
        country=country
    ) if country else REFLECTION_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1
    )

    return get_completion(reflection_prompt, system_message=system_message, client=client)


def one_chunk_improve_translation(
    source_lang,
    target_lang,
    source_text,
    translation_1,
    reflection,
    client = ''
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

    return get_completion(prompt, system_message=system_message, client=client)


def one_chunk_translate_text(
    source_lang, target_lang, source_text, country = "", client = ''
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
        source_lang, target_lang, source_text, client=client
    )
    if translation_1 is None:
        return None

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country, client=client
    )
    if reflection is None:
        return None
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection, client=client
    )
    if translation_2 is None:
        return None
    return {"init_translation": translation_1, "reflection": reflection, "improved_translation": translation_2}


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
    country = "",
    client = ''
) -> str:
    """
    Translate the source_text from source_lang to target_lang.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for the target language.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        str: The improved translation of the source text.
    """
    return one_chunk_translate_text(
        source_lang, target_lang, source_text, country, client=client
    )
