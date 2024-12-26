from translation_agent import translate, split_text
import time
import re
from tqdm import tqdm
import random
from loguru import logger
import concurrent.futures

global_result = dict()

def translate_text(text, index):
    time.sleep(random.randint(20, 40))
    source_lang, target_lang, country = "Chinese", "English", "China"
    logger.info(f'translate {text}')
    translation_triple = translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=text,
        country=country,
    )
    if translation_triple is not None:
        global_result[index] = translation_triple
    return translation_triple


def translate_multiple_thread(text_splits, max_workers=4):
    remained_splits = []
    init_count = len(global_result)
    for i, text in enumerate(text_splits):
        if i not in global_result:
            remained_splits.append([i, text])
    with tqdm(total=len(remained_splits), desc="Translate text") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_translation = {executor.submit(translate_text, pair[1], pair[0]): pair for pair in remained_splits}
            # 提交每个问题的处理
            for future in concurrent.futures.as_completed(future_to_translation):
                (index, text) = future_to_translation[future]
                if not future.result():
                    pbar.update(1)
    logger.info(f'Multi-thread translation done, total: {len(text_splits)}, init: {init_count}, final: {len(text_splits)}')
    return len(global_result) == len(text_splits)


def split_novel(source_text):
    splits = re.split(r'(\n\n第.章)', source_text.replace(u'\u3000', ' '))
    chapters = []
    current = ''
    for split in splits:
        if re.match(r'\n\n第.章', split):
            chapters += split_text(current, 300)
            current = split
        else:
            current = current + split
    chapters += split_text(current, 300)
    return chapters


if __name__ == "__main__":
    source_lang, target_lang, country = "Chinese", "English", "China"

    file_path = '九州·斛珠夫人.chapt1.txt'
    with open(file_path, encoding="utf-8") as file:
        source_text = file.read()

    chapters = split_novel(source_text)

    tries = 0
    MAX_TRIES = 5
    total_time = 0
    while True:
        logger.info(f'Try {tries} time')
        start = time.time()
        status = translate_multiple_thread(chapters)
        end = time.time()
        total_time += end-start
        logger.info(f'Translation done, cost {end-start} seconds. status:{status}')
        if status:
            break
        tries += 1
        if tries >= MAX_TRIES:
            logger.info('exceeed MAX_TRIES {MAX_TRIES}, break')
            break
    status = len(global_result) == len(chapters)
    logger.info(f'Translation done, cost {total_time} seconds, status:{status}')
    with open('init.txt', 'w') as f_init, open('reflection.txt', 'w') as f_ref, open('final.txt', 'w') as f_final:
        for key in sorted(global_result):
            init = global_result[key][0]
            reflect = global_result[key][1]
            final = global_result[key][2]
            f_init.write(init+ "\n")
            f_ref.write(reflect+"\n")
            f_final.write(final)



