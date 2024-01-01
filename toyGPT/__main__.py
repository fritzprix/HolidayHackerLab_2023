from typing import Any
import json
import argparse
import torch
from model import ToyGPT
from data import WikiSourceDataModule
from transformers import GPT2Tokenizer,PreTrainedTokenizer
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb


def get_tokenizer() -> PreTrainedTokenizer:
    
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<pad>", "bos_token":"<|startoftext|>", "eos_token":"<|endoftext|>"}) # special 
    return tokenizer

def get_device() -> Any:
    # will use GPU whenever it's available
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def download(args):
    wikisource_data = WikiSourceDataModule(get_tokenizer(), max_length=510, num_proc=15)
    wikisource_data.prepare_data()


def get_config(path):
    with open(path) as fp:
        return json.load(fp)
    
def train(args):
    
    device = get_device()
    config = get_config(args.config)

    print(f"training performed on {device}")
    torch.set_float32_matmul_precision('high') # enable bfloat16 fast matmul

    # initialize wandb
    wandb.login()
    wandb.init(project="toygpt", config={
        "batch_size": args.batch,
        "learning_rate": args.lr,
        **config
    })

    wandb_logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    
    trainer = L.Trainer(max_epochs=1, val_check_interval=2000, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint('checkpoints', monitor='val_loss', mode='min',filename='model-{epoch}-{val_loss:.3f}', save_top_k=2)
    ],logger=wandb_logger)

    
    tokenizer: PreTrainedTokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    wikisource_data = WikiSourceDataModule(get_tokenizer(), languages=['en'], max_length=config['block_size'], batch_size=args.batch, num_proc=15, train_size=0.99)
    print(f"tokenizer: {tokenizer} / vocab_size {vocab_size} / pad_id:{tokenizer.pad_token_id}, {tokenizer.pad_token}")
    model = ToyGPT(vocab_size=vocab_size, pad_id=tokenizer.pad_token_id, device=device, dtype=torch.float32, dropout=0.2, weight_decay=args.wd, lr=args.lr, **config)
    trainer.fit(model, wikisource_data)
    wandb.finish(0)

    
def generate(args):
    device = get_device()
    tokenizer = get_tokenizer()
    model = ToyGPT.load_from_checkpoint('checkpoints/model-v7.ckpt').to(device)
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

    down_parser = sub_parser.add_parser('download', help='download data')
    down_parser.set_defaults(func=download)
    
    train_parser = sub_parser.add_parser('train', help='train model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-c', '--config', type=str, default='config.json', help='configuration file for training')
    train_parser.add_argument('-b', '--batch', type=int, default=4, help='batch_size for training')
    train_parser.add_argument('-r', '--lr', type=float, default=1e-5, help='learning rate')
    train_parser.add_argument('-d', '--wd', type=float, default=0.01, help='weight decay for Adam optimizer')

    generate_parser = sub_parser.add_parser("generate", help='generate text using model')
    generate_parser.add_argument('-p', '--prompt', type=str, required=True)
    generate_parser.set_defaults(func=generate)
    

    args = arg_parser.parse_args()
    args.func(args)
