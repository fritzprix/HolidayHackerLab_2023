from typing import Any
import torch
import lightning as L

from transformers import PreTrainedTokenizer, GPT2Tokenizer
from data import DataModuleGroup
from model import ToyGPTMLM


def get_tokenizer() -> PreTrainedTokenizer:
    
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                  "mask_token": "<msk>",
                                  "cls_token": "<cls>",
                                  "sep_token": "<sep>",
                                  "bos_token":"<|startoftext|>", 
                                  "eos_token":"<|endoftext|>",}) # special 
    return tokenizer



def get_device() -> Any:
    # will use GPU whenever it's available
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


tokenizer = get_tokenizer()
device = get_device()
batch_size = 8
block_size = 512


from data import HFCollectionDataModule

clm_dataset = HFCollectionDataModule(tokenizer, 
                                            paths=['yahma/alpaca-cleaned'], columns=["output"],
                                            subsets=[[None]], max_length=block_size, batch_size=1)

mlm_dataset = HFCollectionDataModule(tokenizer, 
                                            paths=['yahma/alpaca-cleaned'], columns=["output"],
                                            pretrain_type='MLM',
                                            subsets=[[None]], max_length=block_size, batch_size=1)

data = DataModuleGroup([clm_dataset, mlm_dataset], ["clm", "mlm"], batch_size=batch_size, pad_token_id=tokenizer.pad_token_id)


trainer = L.Trainer(max_epochs=1)

model = ToyGPTMLM(vocab_size=len(tokenizer), name='toygpt_mlm', batch=batch_size, block_size=block_size, n_embed=768, n_head=8, n_layer=12, mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id, device=device)

trainer.fit(model, datamodule=data)

