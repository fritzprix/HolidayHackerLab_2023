from typing import Any, List, Dict

import lightning as L
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import json
from torch.utils import data
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.translate import chunk_by_attention_window
from unstructured.cleaners.core import (
    replace_unicode_quotes, clean, clean_ligatures
)
import re
import json
from unstructured.cleaners.core import clean
import torch

class SentenceChunker:
    """
    A class responsible for chunking text into sentences and tokenizing them
    according to a specified maximum length.

    Attributes:
        tokenizer (PreTrainedTokenizer): A tokenizer from the transformers library
                                         used for tokenizing sentences.
        max_length (int): The maximum token length for a single chunk.
    """

    def _split_into_sentences(self, text):
        """
        Splits the input text into sentences.

        The text is first cleaned to standardize it (removing extra whitespaces, 
        replacing unicode quotes, and removing ligatures). Then, it is split into 
        sentences using a regular expression that looks for sentence end markers 
        (., !, ?) followed by a whitespace.

        Args:
            text (str): The text to be split into sentences.

        Returns:
            List[str]: A list of sentences extracted from the input text.
        """
        # Clean the text and split it into sentences
        clean_text = replace_unicode_quotes(clean_ligatures(clean(text, extra_whitespace=True)))
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        return [f'{sentence}' for sentence in sentences]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int) -> None:
        print('test v1.0')
        """
        Initializes the SentenceChunker with a tokenizer and a maximum length.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_length (int): The maximum token length for a single chunk.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        """
        Processes a batch of text sequences by first splitting them into sentences,
        then encoding each sentence. The sentences are then chunked according to the 
        maximum length, ensuring no chunk exceeds this limit.

        Args:
            batch: A batch of text sequences.

        Returns:
            Dict[str, List]: A dictionary with two keys, 'success' and 'failure'.
                             'success' contains chunks that are within the max_length,
                             'failure' contains chunks that exceed the max_length.
        """
        # Handle single string inputs by wrapping them in a list
        if isinstance(batch, str):
            batch = [batch]


        # Split each sequence in the batch into sentences and encode them
        batch_of_chunks = [self._split_into_sentences(seq) for seq in batch]
        batch_of_encodings = [self.tokenizer.batch_encode_plus(chunks, return_length=True, add_special_tokens=True) for chunks in batch_of_chunks]

        result = {"success": [], "failure": []}
        success_batch_bucket = []
        failure_batch_bucket = []

        # Iterate over each sequence's encodings and chunk them
        for bi, encodings in enumerate(batch_of_encodings):
            bucket = []
            tokens_total = 0

            # Process each sentence in the sequence
            for n, token_count in enumerate(encodings["length"]):
                token_count += 2 # splitting sequence removes space between two adjacent sequence in the process, so 1 token is accounted
                # Handle sentences that exceed the max length
                if token_count > self.max_length:
                    failure_batch_bucket.append({"text":batch_of_chunks[bi][n], "length": token_count})
                    if len(bucket) > 0: # something in the bucket, complete a sequence and start new sequence, because dropping the middle causes discontinuity
                        success_batch_bucket.append({"text":' '.join(bucket), "length": tokens_total})
                        bucket.clear()
                        tokens_total = 0
                    continue

                # Check if adding the sentence would exceed the max length
                if token_count + tokens_total > self.max_length:
                    # Current bucket is full, save and reset it
                    success_batch_bucket.append({"text":' '.join(bucket), "length": tokens_total})
                    bucket.clear()
                    tokens_total = 0
                
                # Add the sentence to the current bucket
                bucket.append(batch_of_chunks[bi][n])
                tokens_total += token_count

            # Append the processed batches to the result
            result["success"].append([*success_batch_bucket])
            result['failure'].append([*failure_batch_bucket])
            success_batch_bucket.clear()
            failure_batch_bucket.clear()

        return result
                
class SuccessCaseGenerator:
    def __init__(self, datasets: List[Dataset], transform=None) -> None:
        self.datasets = datasets
        self.transform = transform

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for ds in self.datasets:
            for b in ds["success"]:
                for seq in b:
                    assert len(seq["text"]) != 0
                    if self.transform:
                        seq = self.transform(seq)
                    yield seq


class WikiSourceDataModule(L.LightningDataModule):

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int, batch_size:int, languages:List[str]=['en'], train_size:float=0.9, num_proc=15) -> None:
        super().__init__()
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.dataset = None


    def prepare_data(self) -> None:

        def transform(v:Dict[str, Any]) -> str:
            return {"length": v["length"], "text": f"<|startoftext|>{v['text']}<|endoftext|>"}
        
        sentence_chunker = SentenceChunker(self.tokenizer, self.max_length - 2)
        datasets = [load_dataset('wikimedia/wikisource', f"20231201.{lang}")["train"].map(lambda b: sentence_chunker(b["text"]), batched=True, num_proc=self.num_proc).flatten() for lang in self.languages]
        success_ds = Dataset.from_generator(SuccessCaseGenerator(datasets, transform=transform))
        success_ds = success_ds.train_test_split(test_size=(1 - self.train_size), train_size=self.train_size)
        success_ds.save_to_disk('local_dscache')

    def setup(self, stage: str) -> None:
        if self.dataset is None:
            self.dataset = load_from_disk('local_dscache')
            self.train_dataset = self.dataset['train'].shuffle().train_test_split(test_size=(1 - self.train_size), train_size=self.train_size)

        return super().setup(stage)

    def _tokenize(self, data):
        inputs = data['text']
        return self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, return_length=True, max_length=self.max_length)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.train_dataset["train"].shuffle().map(self._tokenize, batched=True, batch_size=self.batch_size).select_columns(["input_ids", "attention_mask"])
        return data.DataLoader(train_dataset.with_format(type="torch"), num_workers=self.num_proc)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = self.train_dataset["test"].shuffle().map(self._tokenize, batched=True, batch_size=self.batch_size).select_columns(["input_ids", "attention_mask"])
        return data.DataLoader(val_dataset.with_format(type="torch"), num_workers=self.num_proc)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataset = self.dataset["test"].shuffle().map(self._tokenize, batched=True, batch_size=self.batch_size).select_columns(["input_ids", "attention_mask"])
        return data.DataLoader(test_dataset.with_format(type="torch"), num_workers=self.num_proc)
    

"""https://discuss.huggingface.co/t/map-fails-for-more-than-4-processes/58567/3"""