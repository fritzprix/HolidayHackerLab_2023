import argparse
from datasets import load_dataset
from model import ToyGPT
from data import Preprocessor
from typing import Any
from transformers import GPT2Tokenizer,PreTrainedTokenizer
import os


def process_dataset(args):
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    preprocessor = Preprocessor(tokenizer=tokenizer)
    files = os.listdir(args.data)
    dataset = load_dataset('json', data_files=[os.path.join(args.data, file) for file in files])
    dataset = dataset.map(lambda d: preprocessor(d["text"]))
    print(dataset)
    print(next(dataset['input_ids']).shape)
    dataset.save_to_disk('dataset/train')
    return dataset

def train(args):
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<pad>", "bos_token":"<s>", "eos_token":"</s>"})
    dataset = process_dataset(args)
    
    pass



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('__main__.py')
    arg_parser.set_defaults(func= lambda _: arg_parser.print_help())
    sub_parser = arg_parser.add_subparsers()
    
    train_parser = sub_parser.add_parser('train', help='train model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-c', '--config', type=str, default='train_config.json', help='configuration file for training')
    
    process_parser = sub_parser.add_parser('data', help='process data and build dataset for training')
    process_parser.set_defaults(func=process_dataset)
    process_parser.add_argument('-d', '--data', type=str, default='processed')

    args = arg_parser.parse_args()
    args.func(args)
