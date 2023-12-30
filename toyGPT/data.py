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
    clean_extra_whitespace, replace_unicode_quotes,
    auto_paragraph_grouper, clean,
    group_broken_paragraphs
)
from unstructured.nlp.partition import is_possible_narrative_text
import re
import json
from unstructured.cleaners.core import clean




class SentenceChunker:

    def _split_into_sentences(self, text):
        # Regex pattern to split on sentence-ending punctuation followed by space or end of string
        clean_text = clean(text, extra_whitespace=True, dashes=True, bullets=True)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        return [sentence for sentence in sentences]
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        if isinstance(batch, str):
            batch = [batch]
        batch_of_chunks = [self._split_into_sentences(seq) for seq in batch]
        batch_of_encodings = [self.tokenizer.batch_encode_plus(chunks, return_length=True) for chunks in batch_of_chunks]

        result = {"success": [], "failure": []}
        success_batch_bucket = []
        failure_batch_bucket = []
        for bi, encodings in enumerate(batch_of_encodings):
            bucket = []
            tokens_total = 0
            for n, token_count in enumerate(encodings["length"]):
                if token_count > self.max_length:
                    failure_batch_bucket.append({"text":batch_of_chunks[bi][n], "length": token_count})
                    continue
                if token_count + tokens_total > self.max_length:
                    # bucket is full
                    success_batch_bucket.append({"text":' '.join(bucket), "length": tokens_total})
                    bucket.clear()
                    tokens_total = 0
                    
                bucket.append(batch_of_chunks[bi][n])
                tokens_total += token_count
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
                    if self.transform:
                        seq = self.transform(seq)
                    yield seq


class WikiSourceDataModule(L.LightningDataModule):

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int, batch_size:int, languages:List[str]=['en'], train_size:float=0.9) -> None:
        super().__init__()
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_size = train_size
        self.batch_size = batch_size


    def prepare_data(self) -> None:

        def transform(v:Dict[str, Any]) -> str:
            return {"length": v["length"], "text": f"<s>{v['text']}</s>"}
        
        sentence_chunker = SentenceChunker(self.tokenizer, self.max_length - 2)
        datasets = [load_dataset('wikimedia/wikisource', f"20231201.{lang}")["train"].map(lambda b: sentence_chunker(b["text"]), batched=True, num_proc=8).flatten() for lang in self.languages]
        success_ds = Dataset.from_generator(SuccessCaseGenerator(datasets, transform=transform))
        success_ds = success_ds.train_test_split(test_size=(1 - self.train_size), train_size=self.train_size)
        success_ds.save_to_disk('local_dscache')

    def setup(self, stage: str) -> None:
        self.dataset = load_from_disk('local_dscache')
        return super().setup(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        def tokenize(data):
            inputs = data['text']
            return self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, return_length=True)
        train_dataset = self.dataset["train"].shuffle().map(tokenize, batched=True, batch_size=self.batch_size)
        return data.DataLoader(train_dataset.to_iterable_dataset().with_format(type="torch"), batch_size=self.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        def tokenize(data):
            inputs = data['text']
            return self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, return_length=True)
        test_dataset = self.dataset["test"].shuffle().map(tokenize, batched=True, batch_size=self.batch_size)
        return data.DataLoader(test_dataset.to_iterable_dataset().with_format(type="torch"), batch_size=self.batch_size)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()
    
