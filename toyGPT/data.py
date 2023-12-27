from typing import Any, List

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import json
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.translate import chunk_by_attention_window
from unstructured.cleaners.core import (
    clean_extra_whitespace, replace_unicode_quotes,
    auto_paragraph_grouper, clean,
    group_broken_paragraphs
)
from unstructured.nlp.partition import is_possible_narrative_text
import re
import json



pattern = r'[\u0080-\u00FF\u2026]'

def clean_text(v: str) -> str:
    v = replace_unicode_quotes(v)
    v = clean_extra_whitespace(v)
    v = group_broken_paragraphs(v)
    return v

def save_dataset_to_localfile(datasets: List[Dataset], file, rotation=1024*1024*256):
    rotation_id = 0
    id = 0
    for dataset in datasets:
        source = f"{dataset.info.dataset_name}_{dataset.info.config_name}"
        fp = open(f"{file}_{rotation_id}.json", 'w+t', encoding='utf-8')
        fp.write('[')
        first = True
        ova_len = 0

        for data in tqdm(dataset):
            
            obj = {'id': id, 'title': data['title'], 'text': clean_text(data['text']), 'source': source}
            json_str = json.dumps(obj, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')  # Convert to bytes
            byte_size = len(json_bytes)  # Measure byte size

            if ova_len + byte_size >= rotation:
                fp.write('\n]\n')  # Close the current file
                fp.close()
                rotation_id += 1
                fp = open(f"{file}_{rotation_id}.json", 'w+t', encoding='utf-8')
                fp.write('[')
                first = True
                ova_len = 0
            
            prefix = "\n" if first else ",\n"
            first = False
            fp.write(f"{prefix}{json_str}")
            ova_len += byte_size
            id += 1

        # Handle the end of the dataset
        if ova_len > 0:
            fp.write('\n]\n')
            fp.close()
            first = True



def download(args):
    print(f"args => {args}")
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)
    else:
        for file in os.listdir(args.dir):
            os.remove(os.path.join(args.dir, file))
        
    datasets = [load_dataset(dataset_name, config_name, split='train', streaming=args.stream) for dataset_name, config_name in dataset_configs]
    save_dataset_to_localfile(datasets, f"{args.dir}/data", rotation=args.chunk)


class Preprocessor:

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, data, *args: Any, **kwds: Any) -> Any:
        # data is array of text
        return self.tokenizer.batch_encode_plus(data, padding=True, add_special_tokens=True, return_tensors="pt")




def clean_line(line:str) -> str:
     line = line.encode('utf-8').decode('unicode-escape')
     return re.sub(pattern, '', clean(replace_unicode_quotes(line), bullets=True, dashes=True)).replace(r'[\u201c\u201d]', '"')

def is_valid_line(line: str) -> bool:
    if len(line.split(' ')) < 2:
        return False
    if not is_possible_narrative_text(line):
        return False
    return True


def convert_pdf_to_json(pdf_path:str, output_path:str, max_window_size:int, tokenizer=PreTrainedTokenizer):
    elements = partition_pdf(pdf_path, include_page_breaks=True)
    objs = []
    with open(output_path, 'wt+') as fp:
        raw_text = ''
        for e in elements:
            raw_text += f"{e.text}\n"
        compact_raw = clean_extra_whitespace(auto_paragraph_grouper(group_broken_paragraphs(raw_text)))
        for line in compact_raw.splitlines(keepends=True):
            if not is_valid_line(line):
                continue
            cleaned_line = clean_line(line)
            chunks = chunk_by_attention_window(cleaned_line, tokenizer=tokenizer, max_input_size=max_window_size)
            objs += [{"text": clean_extra_whitespace(c)} for c in chunks]
        json.dump(objs, fp)

def raw_text_to_json(infile: str, outfile: str,  max_window_size:int, tokenizer=PreTrainedTokenizer):
    objs = []
    with open(infile, 'rt', encoding='utf-8') as fp:
        raw_text = fp.read()
        compact_raw = clean_extra_whitespace(auto_paragraph_grouper(group_broken_paragraphs(raw_text)))
        for line in compact_raw.splitlines(keepends=True):
            if not is_valid_line(line):
                continue
            cleaned_line = clean_line(line).replace("\u201c", "oe")
            chunks = chunk_by_attention_window(cleaned_line, tokenizer=tokenizer, max_input_size=max_window_size)
            objs += [{"text": clean_extra_whitespace(c)} for c in chunks]
    with open(outfile, 'wt+') as fp:
        json.dump(objs, fp)


