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
    tokenizer.add_special_tokens({"pad_token": "<pad>", "bos_token":"<s>", "eos_token":"</s>"})
    return tokenizer


def download(args):
    wikisource_data = WikiSourceDataModule(get_tokenizer(), max_length=512)
    wikisource_data.prepare_data()
        

def train(args):
    # will use GPU whenever it's available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"training performed on {device}")
    torch.set_float32_matmul_precision('high') # enable bfloat16 fast matmul

    # initialize wandb
    wandb.login()
    wandb.init(project="toygpt", config={
        "batch_size": args.batch,
        "learning_rate": args.lr,
    })

    wandb_logger = WandbLogger(name='toygpt',version='0.1.0',log_model="all")
    
    trainer = L.Trainer(max_epochs=1, val_check_interval=500, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=3),
        ModelCheckpoint('checkpoints', monitor='val_loss', mode='min',filename='model-{epoch}-{val_loss:.3f}', save_top_k=2)
    ],logger=wandb_logger)

    
    tokenizer: PreTrainedTokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    wikisource_data = WikiSourceDataModule(get_tokenizer(), max_length=512, batch_size=args.batch)
    print(f"tokenizer: {tokenizer} / vocab_size {vocab_size} / pad_id:{tokenizer.pad_token_id}, {tokenizer.pad_token}")
    model = ToyGPT(vocab_size, 768, 12, 12, tokenizer.pad_token_id, device=device, dtype=torch.float32, dropout=0.2, weight_decay=args.wd, lr=args.lr)
    trainer.fit(model, wikisource_data)
    wandb.finish(0)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('__main__.py')
    arg_parser.set_defaults(func= lambda _: arg_parser.print_help())
    sub_parser = arg_parser.add_subparsers()

    down_parser = sub_parser.add_parser('download', help='download data')
    down_parser.set_defaults(func=download)
    
    train_parser = sub_parser.add_parser('train', help='train model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-c', '--config', type=str, default='train_config.json', help='configuration file for training')
    train_parser.add_argument('-b', '--batch', type=int, default=4, help='batch_size for training')
    train_parser.add_argument('-r', '--lr', type=float, default=1e-5, help='learning rate')
    train_parser.add_argument('-d', '--wd', type=float, default=0.01, help='weight decay for Adam optimizer')
    

    args = arg_parser.parse_args()
    args.func(args)
