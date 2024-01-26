from typing import Any, List, Dict
import torch
import shutil

import random
from torch.nn.utils.rnn import pad_sequence

import lightning as L
from datasets import Dataset, load_from_disk, load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizer
from torch.utils import data
from unstructured.cleaners.core import (
    replace_unicode_quotes, clean, clean_ligatures
)
import re
from transformers import PreTrainedTokenizer
import os
from unstructured.cleaners.core import clean

import random

def random_indices(total_elements, portion):
    # Calculate the number of elements to select
    number_to_select = round(total_elements * portion)

    # Generate a list of unique indices for selection
    selected_indices = random.sample(range(total_elements), number_to_select)

    # Calculate the not-selected indices
    all_indices = set(range(total_elements))
    not_selected_indices = list(all_indices - set(selected_indices))

    return selected_indices, not_selected_indices

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

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int, max_sentence_count=None, sep_token=' ', return_failure=False) -> None:
        """
        Initializes the SentenceChunker with a tokenizer and a maximum length.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_length (int): The maximum token length for a single chunk.
        """
        self.max_sentence_count = max_sentence_count
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_failure = return_failure
        self.sep_token = sep_token

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

        result = {"success": []}
        if self.return_failure:
            result.update({"failure": []})
        success_batch_bucket = []
        failure_batch_bucket = []

        # Iterate over each sequence's encodings and chunk them
        for bi, encodings in enumerate(batch_of_encodings):
            bucket = []
            tokens_total = 0

            # Process each sentence in the sequence1
            for n, token_count in enumerate(encodings["length"]):
                token_count += 2 # splitting sequence removes space between two adjacent sequence in the process, so 1 token is accounted
                # Handle sentences that exceed the max length
                if token_count > self.max_length:
                    if self.return_failure:
                        failure_batch_bucket.append({"text": batch_of_chunks[bi][n], "length": token_count})
                    if len(bucket) > 0: # something in the bucket, complete a sequence and start new sequence, because dropping the middle causes discontinuity
                        success_batch_bucket.append({"text": self.sep_token.join(bucket).strip(), "length": tokens_total})
                        bucket.clear()
                        tokens_total = 0
                    continue

                if self.max_sentence_count is not None:
                    if len(bucket) >= self.max_sentence_count:
                        # if the number of setences in the bucket reaches max. 
                        # then add the sentences into success batch
                        success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                        bucket.clear()
                        tokens_total = 0
                        continue

                # Check if adding the sentence would exceed the max length
                if token_count + tokens_total > self.max_length:
                    # Current bucket is full, save and reset it
                    success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                    bucket.clear()
                    tokens_total = 0
                
                # Add the sentence to the current bucket
                bucket.append(batch_of_chunks[bi][n])
                tokens_total += token_count

            if len(bucket) > 0:
                success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                bucket.clear()
                tokens_total = 0
            # Append the processed batches to the result
            result["success"].append([*success_batch_bucket])
            assert len(result["success"]) < self.max_length
            if self.return_failure:
                result["failure"].append([*failure_batch_bucket])
            success_batch_bucket.clear()
            failure_batch_bucket.clear()
        return result


def generate_choices(list_of_indices:List[int], choice_fraction):
    # Shuffle the list to ensure randomness
    k = len(list_of_indices) * choice_fraction
    if k < 1:
        raise ValueError("Choice fraction too small to create any set.")
    list_of_indices = set(list_of_indices)

    k = int(k)
    set_of_choices = []
    while len(list_of_indices) > k:
        choices = set(random.sample(list(list_of_indices), k=k))
        list_of_indices = list_of_indices.difference(choices)
        set_of_choices.append(choices)
    if len(list_of_indices) > 0:
        set_of_choices.append(list_of_indices)
    return set_of_choices

class MLMAugmentation:

    def __init__(self,  datasets: List[Dataset], tokenizer: PreTrainedTokenizer, colunm_selection:str, sep_token_id:int, masking_fraction:float=0.15, ) -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.column_selection = colunm_selection
        self.masking_fraction = masking_fraction
        self.sep_token_id = sep_token_id


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for dataset in self.datasets:
            for data in dataset:
                for sample in data[self.column_selection]:
                    text = sample['text']
                    result = self.tokenizer(f"<cls>{text}<sep>", return_tensors="pt", return_attention_mask=False)
                    input_ids:torch.Tensor = result['input_ids']
                    poplulation = torch.nonzero(input_ids.squeeze() != self.sep_token_id).squeeze().tolist()
                    poplulation.remove(0)
                    choices = generate_choices(poplulation, self.masking_fraction)
                    label:torch.Tensor = input_ids.clone().squeeze(0)
                    input_ids = input_ids.expand((len(choices), input_ids.size(-1)))
                    for i in range(len(choices)):
                        input_ids[i, list(choices[i])] = self.tokenizer.mask_token_id
                        yield {"input": input_ids[i], "label": label}
    
class CLMAugmentation:

    def __init__(self, datasets: List[Dataset], tokenizer: PreTrainedTokenizer, colunm_selection:str) -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.column_selection = colunm_selection

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        for dataset in self.datasets:
            for data in dataset:
                for sample in data[self.column_selection]:
                    assert sample['length'] < 512
                    text = sample['text']
                    result = self.tokenizer(f"<|startoftext|>{text}<|endoftext|>", return_attention_mask=False)
                    input_ids = result["input_ids"]
                    yield {"input": input_ids[:-1], "label": input_ids[1:]}
                
    



class HuggingFaceCollectionModuleV1(L.LightningDataModule):

    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 paths:List[str],
                 subsets: List[List[str]],
                 columns: List[str],
                 max_length:int,
                 batch_size:int,
                 pretrain_type:str='CLM',
                 clear_cache:bool=False,
                 train_size:float=0.9,
                 resume_pos = 0,
                 num_proc=15) -> None:
        super().__init__()

        self.name = '_'.join(paths)
        self.tokenizer = tokenizer
        self.paths = paths
        self.subsets = subsets
        self.columns = columns
        self.max_length = max_length
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.clear_cache = clear_cache
        self.resume_pos = resume_pos
        self.dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.pretrain_type = pretrain_type
        self.local_fdata_cache_path = f'cache/{self.name}_{pretrain_type}/local_dscache'
        self.local_tdata_cache_path = f'cache/{self.name}_{pretrain_type}/train_dscache'
        self.local_vdata_cache_path = f'cache/{self.name}_{pretrain_type}/val_dscache'
        self.local_tokenized_cache_path = f'cache/{self.name}_{pretrain_type}/tokenized'


    def prepare_data(self) -> None:
        full_dataset = None
        
        if self.clear_cache:
            shutil.rmtree(self.local_fdata_cache_path, ignore_errors=True)
            shutil.rmtree(self.local_tdata_cache_path, ignore_errors=True)
            shutil.rmtree(self.local_vdata_cache_path, ignore_errors=True)
        
        if not os.path.exists(self.local_fdata_cache_path):
            

            sentence_chunker = SentenceChunker(self.tokenizer, self.max_length - 2, sep_token=' ' if self.pretrain_type == 'CLM' else '<sep>')
            datasets = [load_dataset(path, subset, num_proc=self.num_proc)['train']
                        .map(lambda b: sentence_chunker(b[column]), batched=True, num_proc=self.num_proc).flatten().select_columns(['success']) for i, (path, column) in enumerate(zip(self.paths, self.columns)) for subset in self.subsets[i]]
            if self.pretrain_type == 'CLM':
                preprocessor = CLMAugmentation(datasets, self.tokenizer, colunm_selection="success")
            elif self.pretrain_type == 'MLM':
                preprocessor = MLMAugmentation(datasets, self.tokenizer, colunm_selection="success", sep_token_id=self.tokenizer.sep_token_id)
            print(preprocessor)
            full_dataset = Dataset.from_generator(preprocessor, num_proc=self.num_proc)
            print(full_dataset)
            full_dataset = full_dataset.train_test_split(test_size=(1 - self.train_size), train_size=self.train_size)
            full_dataset.save_to_disk(self.local_fdata_cache_path, num_proc=self.num_proc)
        else:
            print('full data cached locally')
        
        if not (os.path.exists(self.local_tdata_cache_path) and os.path.exists(self.local_vdata_cache_path)):
            if full_dataset is None:
                full_dataset = load_from_disk(self.local_fdata_cache_path)
            visible_dataset = full_dataset['train'].shuffle()
            val_selection, train_selection = random_indices(len(visible_dataset), (1 - self.train_size))
            val_dataset = visible_dataset.select(val_selection)
            train_dataset = visible_dataset.select(train_selection)
            val_dataset.save_to_disk(self.local_vdata_cache_path, num_proc=self.num_proc)
            train_dataset.save_to_disk(self.local_tdata_cache_path, num_proc=self.num_proc)
        else:
            print('load from local cache')

    def setup(self, stage: str) -> None:
        if self.dataset is None:
            self.dataset = load_from_disk(self.local_fdata_cache_path)
            self.val_dataset = load_from_disk(self.local_vdata_cache_path)
            self.train_dataset = load_from_disk(self.local_tdata_cache_path)

        return super().setup(stage)
    
    
    def _prepare_for_model(self, data):

        # Pad input sequences
        inputs_padded = pad_sequence([seq['input'].long() for seq in data], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        batch_size = inputs_padded.size(0)

        # For CLM, create a triangular attention mask (lower triangular matrix)
        if self.pretrain_type == 'CLM':
            max_len = inputs_padded.size(1)
            attention_masks = torch.tril(torch.ones((max_len, max_len), dtype=torch.long)).expand((batch_size, max_len, max_len))
        elif self.pretrain_type == 'MLM':
            attention_masks = (inputs_padded != self.tokenizer.pad_token_id).int()

        targets_padded = pad_sequence([tgt['label'].long() for tgt in data], batch_first=True, padding_value=self.tokenizer.pad_token_id)


        return {
            'input': inputs_padded,
            'target': targets_padded,
            'attention_mask': attention_masks
        }


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.train_dataset
        l = len(train_dataset)
        print(range(self.batch_size * self.resume_pos, l))
        
        train_dataset = train_dataset.select(range(self.batch_size * self.resume_pos, l)).select_columns(["input", "label"])
        return data.DataLoader(train_dataset.with_format(type="torch"),  batch_size=self.batch_size, collate_fn=self._prepare_for_model)
    
    #.skip(self.batch_size * self.resume_pos)
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset:Dataset = self.val_dataset.select_columns(["input", "label"])
        return data.DataLoader(val_dataset.with_format(type="torch"), batch_size=self.batch_size, collate_fn=self._prepare_for_model)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataset = self.dataset["test"].select_columns(["input", "label"])
        return data.DataLoader(test_dataset.with_format(type="torch"),  batch_size=self.batch_size, collate_fn=self._prepare_for_model)
    

