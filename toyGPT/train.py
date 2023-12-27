import argparse

import lightning as L
import torch
from datasets import load_dataset
from model import ToyGPT
from data import Preprocessor
from transformers import AutoTokenizer

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    dataset = load_dataset('json', data_dir='raw')
    
    print('hello')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('train.py')
    main(argparser.parse_args())