from difflib import Differ
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import docx
import gradio as gr
import pymupdf
from icecream import ic
from patch import (
    model_load,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
    translate_whole_text,
    CLIENT
)
from simplemma import simple_tokenizer

from loguru import logger
progress = gr.Progress()


def extract_text(path):
    with open(path) as f:
        file_text = f.read()
    return file_text


def extract_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_docx(path):
    doc = docx.Document(path)
    data = []
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)
    content = "\n\n".join(data)
    return content


def tokenize(text):
    # Use nltk to tokenize the text
    words = simple_tokenizer(text)
    # Check if the text contains spaces
    if " " in text:
        # Create a list of words and spaces
        tokens = []
        for word in words:
            tokens.append(word)
            if not word.startswith("'") and not word.endswith(
                "'"
            ):  # Avoid adding space after punctuation
                tokens.append(" ")  # Add space after each word
        return tokens[:-1]  # Remove the last space
    else:
        return words


def diff_texts(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    d = Differ()
    diff_result = list(d.compare(tokens1, tokens2))

    highlighted_text = []
    for token in diff_result:
        word = token[2:]
        category = None
        if token[0] == "+":
            category = "added"
        elif token[0] == "-":
            category = "removed"
        elif token[0] == "?":
            continue  # Ignore the hints line

        highlighted_text.append((word, category))

    return highlighted_text


# modified from src.translaation-agent.utils.tranlsate
def translator(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
):
    """Translate the source_text from source_lang to target_lang."""
    return translator_sec(endpoint2="", base2="", model2="", api_key2="", source_lang=source_lang, target_lang=target_lang, source_text=source_text, country=country)


def translator_sec(
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
):
    """Translate the source_text from source_lang to target_lang."""
    init_translation = ''
    reflection = ''
    final_translation = ''
     
    final_result = translate_whole_text(source_text, source_lang, target_lang, country, 64, client=CLIENT)
    logger.info(f'final result: {final_result}')
    for segment in final_result:
        logger.info(f'segment {segment}')
        if 'status' in segment and segment['status'] == 'success':
            init_translation += segment['init_translation'] + "\n"
            reflection += segment['reflection'] + "\n"
            final_translation += segment['improved_translation'] + "\n"
        else:
            init_translation += "failed\n"
            reflection += "failed\n"
            final_translation += "failed\n"

    logger.info(f'translate: {source_text}')
    logger.info(f'init translation: {init_translation}')
    logger.info(f'reflection: {reflection}')
    logger.info(f'final translation: {final_translation}')
    return init_translation, reflection, final_translation
