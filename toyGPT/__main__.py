from typing import Any
import json
import argparse
import torch
import re
from model import ToyGPT
from data import WikiSourceDataModule, RedPajamaDataModule, RedPajamaDataSampleModule, HuggingFaceCollectionModule
from transformers import GPT2Tokenizer,PreTrainedTokenizer
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import os


def get_last_file(dir_path: str) -> str:
    files = [os.path.join(dir_path,fname) for fname in os.listdir(dir_path)]
    files.sort(key=lambda x: os.path.getatime(x), reverse=True)
    if files:
        return files[0]
    else:
        return None

def get_tokenizer() -> PreTrainedTokenizer:
    
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<pad>", "bos_token":"<|startoftext|>", "eos_token":"<|endoftext|>"}) # special 
    return tokenizer

def get_device() -> Any:
    # will use GPU whenever it's available
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


def get_steps(model_name:str) -> int:
    step_number = re.search(r"step=(\d+)", model_name)
    step_number = int(step_number.group(1)) if step_number else None
    return step_number
    
def train(args):
    device = get_device()
    print(f"training will be performed on {device}")
    configs = get_config(args.config)
    config = configs[0]


    # initialize wandb
    if args.wnb:
        wandb.login()
        wandb.init(project="toygpt", config={
            "batch_size": args.batch,
            "learning_rate": args.lr,
            **config
        })

    if args.wnb:
        logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    else:
        logger = TensorBoardLogger('tf_logs')
    # 
    trainer = L.Trainer(max_epochs=1,  precision=args.precision, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint('checkpoints', monitor='val_loss', mode='min',filename='model-{step}-{val_loss:.3f}', save_top_k=2)
    ],val_check_interval=2000, logger=logger)

    
    tokenizer: PreTrainedTokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    data_module = HuggingFaceCollectionModule(tokenizer, 
                                              paths=['wikimedia/wikisource', "togethercomputer/RedPajama-Data-1T-Sample"],
                                              subsets=[
                                                  ['20231201.en'],
                                                  [None]
                                              ],
                                              max_length=config['block_size'], 
                                              batch_size=args.batch, 
                                              num_proc=15, 
                                              train_size=0.99)
    
    dtype = get_dtype(args.precision)
    
    print(f"tokenizer: {tokenizer} / vocab_size {vocab_size} / pad_id:{tokenizer.pad_token_id}, {tokenizer.pad_token}")
    model = ToyGPT(vocab_size=vocab_size, pad_id=tokenizer.pad_token_id, dtype=dtype, device=device, p_dropout=0.1, weight_decay=args.wd, lr=args.lr, batch=args.batch, **config)
    
    trainer.fit(model, data_module)
    if args.wnb:
        wandb.finish(0)

def process(args):
    config = get_config(args.config)
    
                                           
    wikisource_data = WikiSourceDataModule(get_tokenizer(), 
                                           languages=['en'], 
                                           max_length=config['block_size'], 
                                           clear_cache=True, 
                                           batch_size=args.batch, 
                                           num_proc=15, 
                                           train_size=0.99)
    wikisource_data.prepare_data()


def resume(args):
    if args.wnb:
        logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    else:
        logger = TensorBoardLogger('tf_logs')
    tokenizer = get_tokenizer()
    device = get_device()
    torch.set_float32_matmul_precision("medium")
    last_ckpt_name = get_last_file('checkpoints')
    model = ToyGPT.load_from_checkpoint(last_ckpt_name, device=device)
    print(model.hparams)
    batch_size = model.hparams['batch']
    block_size = model.hparams['block_size']
    trainer = L.Trainer(max_epochs=1,  precision=args.precision, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint('checkpoints', monitor='val_loss', mode='min',filename='model-{step}-{val_loss:.3f}', save_top_k=2)
    ],val_check_interval=2000, logger=logger)
    
    steps = get_steps(last_ckpt_name)
    print(f"resusmed state : {last_ckpt_name}  (steps: {steps})")
    print(f"hparam: \n {model.hparams})")
    data_module = HuggingFaceCollectionModule(tokenizer, paths=['wikimedia/wikisource', "togethercomputer/RedPajama-Data-1T-Sample"],
                                              subsets=[
                                                  ['20231201.en'],
                                                  [None]
                                              ],
                                              max_length=block_size, 
                                              batch_size=batch_size, 
                                              resume_pos=steps,
                                              num_proc=15, 
                                              train_size=0.99)
    trainer.fit(model, data_module)
    
    
    
def generate(args):
    device = get_device()
    tokenizer = get_tokenizer()
    model = ToyGPT.load_from_checkpoint('checkpoints/model-v8.ckpt').to(device)
    model.eval()

    prompt = f"{tokenizer.bos_token}{args.prompt}"
    input = tokenizer(prompt, return_attention_mask=True, return_tensors="pt").to(device)

    for _ in range(300):
        output = model(input)  # Assuming the model returns logits
        next_token_id = torch.argmax(output, dim=-1).item()  # Get the most probable next token ID

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = input["input_ids"]
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
    train_parser.add_argument('-d', '--wd', type=float, default=0.1, help='weight decay for Adam optimizer')
    train_parser.add_argument('-p', '--precision', type=str, default='32-true', help='training precision option')
    train_parser.add_argument('-w', '--wnb', type=bool, default=False, help='wandb logging')

    resume_parser = sub_parser.add_parser('resume', help='resume training')
    resume_parser.add_argument('-i', '--ckpt', required=False, default=None)
    resume_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    resume_parser.add_argument('-w', '--wnb', type=bool, default=False, help='wandb logging')
    resume_parser.add_argument('-p', '--precision', type=str, default='32-true', help='training precision option')
    resume_parser.set_defaults(func=resume)

    generate_parser = sub_parser.add_parser("generate", help='generate text using model')
    generate_parser.add_argument('-p', '--prompt', type=str, required=True)
    generate_parser.set_defaults(func=generate)

    process_parser = sub_parser.add_parser('preprocess', help='preprocess')
    process_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    process_parser.add_argument('-b', '--batch', type=int, default=8, help='batch_size for data processing')
    process_parser.set_defaults(func=process)
    

    args = arg_parser.parse_args()
    args.func(args)
