from difflib import Differ

import docx
import gradio as gr
import pymupdf
from icecream import ic
from patch import (
    model_load,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
)
from simplemma import simple_tokenizer


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
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""


    ic("Translating text as single chunk")

    progress((1, 3), desc="First translation...")
    init_translation = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    progress((2, 3), desc="Reflection...")
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, init_translation, country
    )

    progress((3, 3), desc="Second translation...")
    final_translation = one_chunk_improve_translation(
        source_lang, target_lang, source_text, init_translation, reflection
    )

    return init_translation, reflection, final_translation


def translator_sec(
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    ic("Translating text as single chunk")

    progress((1, 3), desc="First translation...")
    init_translation = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    try:
        model_load(endpoint2, base2, model2, api_key2)
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}") from e

    progress((2, 3), desc="Reflection...")
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, init_translation, country
    )

    progress((3, 3), desc="Second translation...")
    final_translation = one_chunk_improve_translation(
        source_lang, target_lang, source_text, init_translation, reflection
    )

    return init_translation, reflection, final_translation
