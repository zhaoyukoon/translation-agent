from translation_agent import translate_whole_text
import time
from loguru import logger
import json

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='article translation')
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-s', '--source_lang', default='Chinese')
    parser.add_argument('-t', '--target_lang', default='English')
    parser.add_argument('-c', '--country', default='China')
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-w', '--workers', default=64, )
    return parser



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    file_path = args.input_file
    with open(file_path, encoding="utf-8") as file:
        source_text = file.read()

    final_result = translate_whole_text(source_text, args.source_lang, args.target_lang, args.country, args.workers)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
