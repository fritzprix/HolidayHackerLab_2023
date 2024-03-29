from typing import Any,List
import json
import argparse
import torch
import re
from model import ToyGPT
from data import HFCollectionMultiTaskDataModule
from transformers import GPT2TokenizerFast,PreTrainedTokenizer, BertTokenizerFast
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import os

def get_checkpoint_path(model_name:str):
    return os.path.join('checkpoints', model_name)

def get_last_file(dir_path: str) -> str:
    files = [os.path.join(dir_path,fname) for fname in os.listdir(dir_path)]
    files.sort(key=lambda x: os.path.getatime(x), reverse=True)
    if files:
        return files[0]
    else:
        return None

def get_tokenizer() -> PreTrainedTokenizer:
    
    tokenizer: PreTrainedTokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                  "mask_token": "<msk>",
                                  "cls_token": "<cls>",
                                  "sep_token": "<sep>",
                                  "bos_token":"<|startoftext|>", 
                                  "eos_token":"<|endoftext|>",}) # special 
    return tokenizer

def get_device() -> Any:
    # will use GPU whenever it's available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_config(path):
    with open(path) as fp:
        return json.load(fp)
    
def get_dtype(precision) -> torch.dtype:
    if precision == '16-true':
        return torch.float16
    if precision == 'bf16-true':
        return torch.bfloat16
    if precision == '16-mixed':
        return torch.float
    if precision == 'bf16-mixed':
        return torch.float
    if precision == '32-true':
        return torch.float

def get_steps(model_name: str) -> int:
    # Extract the offset value
    offset_match = re.search(r"offset=(\d+)", model_name)
    offset = int(offset_match.group(1)) if offset_match else 0  # Default to 0 if not found

    # Extract the step number
    step_match = re.search(r"step=(\d+)", model_name)
    step = int(step_match.group(1)) if step_match else 0  # Default to 0 if not found

    # Calculate the total offset
    total_offset = offset + step
    return total_offset


def train(args):
    device = get_device()
    print(f'Args => {args}')
    print(f"training will be performed on {device}")
    configs = get_config(args.config)
    config = configs[0]
    torch.set_float32_matmul_precision('medium')
    

    # initialize wandb

    if args.wnb:
        wandb.login()
        wandb.init(project="toygpt", config={
            "batch_size": args.batch,
            "learning_rate": args.lr,
            **config
        })
        logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    else:
        logger = TensorBoardLogger('tf_logs')
    

    
    tokenizer: PreTrainedTokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    cpu_count = (os.cpu_count() - 1)

    dataset = HFCollectionMultiTaskDataModule(tokenizer, 
                                            paths=['wikimedia/wikisource', "togethercomputer/RedPajama-Data-1T-Sample"],
                                            subsets=[
                                                  ['20231201.en'],
                                                  [None]
                                            ],
                                            columns=[
                                                  'text','text'
                                            ],
                                            tasks=['CLM'],
                                            cache_dir=args.cache,
                                            max_length=config['block_size'], 
                                            num_proc=cpu_count,
                                            batch_size=args.batch, train_size=0.99)
    dataset.prepare_data()
    train_steps, _ = dataset.setup()
    print(f'total train steps : {train_steps}')
    dtype = get_dtype(args.precision)
    
    print(f"tokenizer: {tokenizer} / vocab_size {vocab_size} / pad_id:{tokenizer.pad_token_id}, {tokenizer.pad_token}")
    model = ToyGPT(vocab_size=vocab_size, 
                      pad_token_id=tokenizer.pad_token_id, dtype=dtype, device=device, 
                      p_dropout=0.1, weight_decay=args.wd, lr=args.lr, batch=args.batch, 
                      **config)

    trainer = L.Trainer(max_epochs=1, log_every_n_steps=args.batch, precision=args.precision, max_steps=train_steps, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint(get_checkpoint_path(model.__class__.__name__), monitor='val_loss', mode='min',filename='model-offset=0-{step}-{val_loss:.3f}', save_top_k=2, save_last=True)
    ], val_check_interval=0.01, logger=logger)

    trainer.fit(model, 
                train_dataloaders=dataset.train_dataloader(), 
                val_dataloaders=dataset.val_dataloader())
    if args.wnb:
        wandb.finish(0)

def process(args):
    config = get_config(args.config)
    tokenizer = get_tokenizer()
    cpu_count = (os.cpu_count() - 1)
    dataset = HFCollectionMultiTaskDataModule(tokenizer, 
                                            paths=['wikimedia/wikisource', "togethercomputer/RedPajama-Data-1T-Sample"],
                                            subsets=[
                                                  ['20231201.en'],
                                                  [None]
                                            ],
                                            columns=[
                                                  'text','text'
                                            ],
                                            tasks=['CLM'],
                                            cache_dir=args.cache, 
                                            num_proc=cpu_count,
                                            max_length=config['block_size'], 
                                            batch_size=args.batch,  train_size=0.99)
    dataset.prepare_data()


def resume(args):
    print(f'Args => {args}')
    tokenizer = get_tokenizer()
    device = get_device()
    print(f"training will be performed on {device}")
    
    torch.set_float32_matmul_precision('medium')
    last_ckpt_name = get_last_file(get_checkpoint_path(ToyGPT.__name__))
    step_offset = get_steps(last_ckpt_name)
    model = ToyGPT.load_from_checkpoint(last_ckpt_name, device=device)
    
    
    if args.wnb:
        wandb.init(project="toygpt")
        logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    else:
        logger = TensorBoardLogger('tf_logs')

    print(model.hparams)
    
    batch_size = model.hparams['batch']
    block_size = model.hparams['block_size']
    
    
    
    print(f"resusmed state : {last_ckpt_name}  (steps: {step_offset})")
    print(f"hparam: \n {model.hparams})")

    cpu_count = (os.cpu_count() - 1)
    dataset = HFCollectionMultiTaskDataModule(tokenizer, 
                                            paths=['wikimedia/wikisource', "togethercomputer/RedPajama-Data-1T-Sample"],
                                            subsets=[
                                                  ['20231201.en'],
                                                  [None]
                                            ],
                                            columns=[
                                                  'text','text'
                                            ],
                                            tasks=['CLM'],
                                            cache_dir=args.cache, 
                                            max_length=block_size, 
                                            num_proc=cpu_count,
                                            batch_size=batch_size, train_size=0.99)
    
    dataset.prepare_data()
    train_steps, _ = dataset.setup()
    print(f'total train steps : {train_steps}')
    trainer = L.Trainer(max_epochs=1, max_steps=train_steps, log_every_n_steps=batch_size, precision=args.precision, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint(get_checkpoint_path(model.__class__.__name__), monitor='val_loss', mode='min',filename=f"model-offset={step_offset}" + '-{step}-{val_loss:.3f}', save_top_k=2, save_last=True)
    ],val_check_interval=0.01, logger=logger)
    trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader(), ckpt_path='last')
    if args.wnb:
        wandb.finish()
    
def apply_repeat_penalty(logits:torch.Tensor, input_ids, penalty_factor):
    new_ids = logits.argmax(dim=-1)
    for i, (new_id, seq) in enumerate(zip(new_ids, input_ids)):
        if new_id in seq:
            logits[i, new_id] *= penalty_factor
    return logits

    
def generate(args):
    device = get_device()
    tokenizer = get_tokenizer()
    if args.model is None:
        model_checkpoint = get_last_file(get_checkpoint_path(ToyGPT.__name__))
    else:
        model_checkpoint = args.model
    model = ToyGPT.load_from_checkpoint(model_checkpoint, device=device)
    model.eval()

    prompt = f"{tokenizer.bos_token}{args.prompt}"
    input = tokenizer(prompt, return_attention_mask=True, return_tensors="pt").to(device)

    for _ in range(300):
        input_ids = input["input_ids"]
        logits = model(input)  # Assuming the model returns logits
        if args.repeat_penalty:
            logits = apply_repeat_penalty(logits=logits, input_ids=input_ids, penalty_factor=1/pow(10, args.repeat_penalty))
        next_token_id = torch.argmax(logits, dim=-1).item()  # Get the most probable next token ID

        if next_token_id == tokenizer.eos_token_id:
            break

        
        new_input_ids = torch.cat((input_ids, torch.tensor([[next_token_id]], device=device)), dim=1)
        new_attention_mask = torch.ones((1, new_input_ids.shape[-1]), device=device)

        input = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}

    generated_text = tokenizer.decode(input['input_ids'][0], skip_special_tokens=True)
    print(generated_text)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('__main__.py')
    arg_parser.set_defaults(func= lambda _: arg_parser.print_help())
    sub_parser = arg_parser.add_subparsers()

    train_parser = sub_parser.add_parser('train', help='train model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    train_parser.add_argument('-b', '--batch', type=int, default=4, help='batch_size for training')
    train_parser.add_argument('-r', '--lr', type=float, default=2.5e-4, help='learning rate')
    train_parser.add_argument('-x', '--cache', type=str, help='path to store local training dataset')
    train_parser.add_argument('-d', '--wd', type=float, default=0.1, help='weight decay for Adam optimizer')
    train_parser.add_argument('-p', '--precision', type=str, default='32-true', help='training precision option')
    train_parser.add_argument('-w', '--wnb', type=bool, default=False, help='wandb logging')

    resume_parser = sub_parser.add_parser('resume', help='resume training')
    resume_parser.add_argument('-i', '--ckpt', required=False, default=None)
    resume_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    resume_parser.add_argument('-w', '--wnb', type=bool, default=False, help='wandb logging')
    resume_parser.add_argument('-p', '--precision', type=str, default='32-true', help='training precision option')
    resume_parser.add_argument('-x', '--cache', type=str, help='path to store local training dataset')
    resume_parser.set_defaults(func=resume)

    generate_parser = sub_parser.add_parser("generate", help='generate text using model')
    generate_parser.add_argument('-p', '--prompt', type=str, required=True)
    generate_parser.add_argument('-m', '--model', type=str, default=None)
    generate_parser.add_argument('-r', '--repeat_penalty', type=float, default=1.3)
    generate_parser.set_defaults(func=generate)

    process_parser = sub_parser.add_parser('preprocess', help='preprocess')
    process_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    process_parser.add_argument('-b', '--batch', type=int, default=8, help='batch_size for data processing')
    process_parser.add_argument('-x', '--cache', type=str, help='path to store local training dataset')
    process_parser.set_defaults(func=process)
    

    args = arg_parser.parse_args()
    args.func(args)
