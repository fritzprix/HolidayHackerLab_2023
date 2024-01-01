from typing import Any, Tuple, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import math
import lightning as L

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, d_model, device=None, dtype: torch.dtype=torch.float32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dk = torch.sqrt(torch.scalar_tensor(d_model, device=device, dtype=dtype))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input should have (B,N,d_model)
        # q (b,1,d_model) , k (b,n,d_model)
        # qk = (b,1,n)
        scaled_qk = q@k.transpose(2, 1) * (1 / math.sqrt(k.size(-1)))
        if mask is not None:
            masked_scaled_qk = scaled_qk.masked_fill(mask=mask.bitwise_not(), value=float('-inf'))
        attention_weights = torch.softmax(masked_scaled_qk, dim=-1)
        return  attention_weights @  v
        


class MultiHeadAttentionV2(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.dropout = dropout

        self.attn_proj = torch.nn.Sequential(torch.nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype),
                                             torch.nn.Dropout(self.dropout))
        
        self.out_linear = torch.nn.Linear(d_model, d_model)

        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        
        B,n_seq,C = input.size()
        q ,k, v = self.attn_proj(input).split(C, dim=-1)
        
        q = q.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        k = k.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        v = v.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2)

        scaled_dot_product = q@k.transpose(-1,-2) * (1 / math.sqrt(k.size(-1))) # (B, n_h, n_seq, n_seq)
        if mask is not None:
            # shape of given mask (B, n_seq, n_seq) we have to unsqueeze to get (B, 1, n_seq,n_seq) so it can be broadcast to (B,n_head, n_seq,n_seq)
            scaled_dot_product = scaled_dot_product.masked_fill(mask=mask.unsqueeze(1).bitwise_not(), value=float('-inf'))
        sdp_out: torch.Tensor = scaled_dot_product.softmax(dim=-1) @ v # (B, n_h, n_seq, d_h)

        return self.out_linear(sdp_out.transpose(1,2).contiguous().view(B,n_seq, C))



class MultiHeadAttentionV1(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.depth = d_model // n_head
        self.dropout = dropout

        self.q_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))
        self.k_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))
        self.v_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))

        self.attns = torch.nn.ModuleList([ScaledDotProductAttention(d_model=self.depth, device=device, dtype=dtype) for _ in range(n_head)])
        self.output_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_drop = torch.nn.Dropout(dropout)

        
        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        b,n,_ = input.shape
        q = self.q_linear(input).view((b, n, self.n_head, -1))
        k = self.k_linear(input).view((b, n, self.n_head, -1))
        v = self.v_linear(input).view((b, n, self.n_head, -1))


        attn_output = torch.concat([self.attns[i].forward(q[:,:,i,:].view((b,n,self.depth)), 
                                            k[:,:,i,:].view((b,n, self.depth)), 
                                            v[:,:,i,:].view((b,n, self.depth)),mask=mask) for i in range(self.n_head)],dim=-1)
        return self.out_drop(self.output_linear(attn_output))
        


class PositionWiseFeedforward(torch.nn.Module):

    def __init__(self, d_model:int, device, dtype: torch.dtype=torch.float32, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pwff = torch.nn.Sequential(torch.nn.Linear(d_model, d_model * 4, device=device,dtype=dtype), 
                                            torch.nn.GELU(), 
                                            torch.nn.Linear(4* d_model, d_model, device=device, dtype=dtype), 
                                            torch.nn.Dropout(dropout))
        
    def forward(self, input: torch.Tensor)-> torch.Tensor:
        return self.pwff.forward(input)


class Transformer(torch.nn.Module):

    def __init__(self, n_head, d_model, device, dtype:torch.dtype=torch.float32,dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_norm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttentionV2(d_model=d_model, n_head=n_head, device=device, dtype=dtype, dropout=dropout)
        self.mha_lnorm = torch.nn.LayerNorm(d_model, device=device,dtype=dtype)
        self.pw_ff = PositionWiseFeedforward(d_model=d_model, device=device, dtype=dtype, dropout=dropout)

    def forward(self, data:Tuple[torch.Tensor]) -> torch.Tensor:
        # Pre-LayerNormalization from GPT-3, (note: Post-LayerNormalization is used for GPT-2 and original paper)
        input, attention_mask = data

        norm_input = self.input_norm.forward(input)
        mha_output = input + self.mha.forward(norm_input, attention_mask)
        norm_mha_output = self.mha_lnorm(mha_output)
        return (mha_output + self.pw_ff.forward(norm_mha_output), attention_mask)
    

class ToyGPT(L.LightningModule):

    def __init__(self, 
                 vocab_size:int, 
                 block_size:int,
                 n_embed:int, n_head:int, n_layer:int, pad_id:int=None,  device=None, 
                 dtype:torch.dtype=torch.float32, dropout:float=0.2, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, name: str='toygpt', *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['dtype', 'device'])
        self.name = name
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.decay = weight_decay
        self.embedding = torch.nn.Embedding(vocab_size, n_embed, padding_idx=pad_id, device=device, dtype=dtype)
        self.pos_embedding = torch.nn.Embedding(block_size, n_embed, device=device, dtype=dtype)
        self.transformers = torch.nn.Sequential(*[Transformer(n_head=n_head, d_model=n_embed, device=device, dtype=dtype, dropout=dropout) for _ in range(n_layer)])
        self.output_linear = torch.nn.Linear(n_embed, vocab_size, device=device, dtype=dtype)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.decay)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=self.eps, total_iters=2000, end_factor=1),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4000, eta_min=(self.lr / 10))
        ],milestones=[2000])
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "CosineWithWarmUp",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        # X should have shape of (B,N)
        X: torch.Tensor = input["input_ids"]
        attention_mask: torch.Tensor = input["attention_mask"]
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        X_wemb = self.embedding(X) + self.pos_embedding(torch.arange(0, X.shape[-1],device=X.device, dtype=torch.long)) # word embedding + postion embedding
        hs, _ = self.transformers.forward((X_wemb, attention_mask.bool()))
        return torch.softmax(self.output_linear.forward(hs[:,-1,:]), -1)


    def training_step(self, data: Tuple[torch.Tensor], batch_index:Any, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        X = data["input_ids"]

        input:torch.Tensor = X[:,:-1]
        target:torch.Tensor = X[:,1:].long()
        B,n = input.shape

        attention_mask:torch.Tensor = torch.tril(torch.ones((n,n), device=input.device)).unsqueeze(0).expand((B,n,n)).bool()
        

        X_wemb = self.embedding(input) + self.pos_embedding(torch.arange(0, input.shape[-1],device=input.device, dtype=torch.long)) # word embedding + postion embedding
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        # the sequencess of batch are now totally flatten into (B * n, logits), so we have to divide the loss by batch_size
        loss = self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1))
        if batch_index % 10 == 0:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            # log train loss not too much frequently
            self.log("train_loss", loss)
            self.log("lr", lr)


        return {"batch_index": batch_index, "loss":loss}
    

    def validation_step(self, data: Tuple[torch.Tensor], batch_index,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        X = data["input_ids"]

        input:torch.Tensor = X[:,:-1]
        target:torch.Tensor = X[:,1:].long()
        B,n = input.shape

        attention_mask:torch.Tensor = torch.tril(torch.ones((n,n), device=input.device)).unsqueeze(0).expand((B,n,n)).bool()

        X_wemb = self.embedding(input) + self.pos_embedding(torch.arange(0, input.shape[-1],device=input.device, dtype=torch.long)) # word embedding + postion embedding
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        # the sequencess of batch are now totally flatten into (B * n, logits), so we have to divide the loss by batch_size
        loss = self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss)
        return {"batch_index": batch_index, "val_loss":loss}
    
    
    def test_step(self, data: Tuple[torch.Tensor], batch_index, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
       
        X = data["input_ids"]

        input:torch.Tensor = X[:,:-1]
        target:torch.Tensor = X[:,1:].long()
        B,n = input.shape

        attention_mask:torch.Tensor = torch.tril(torch.ones((n,n), device=input.device)).unsqueeze(0).expand((B,n,n)).bool()

        X_wemb = self.embedding(input) + self.pos_embedding(torch.arange(0, input.shape[-1],device=input.device, dtype=torch.long)) # word embedding + postion embedding
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        # the sequencess of batch are now totally flatten into (B * n, logits), so we have to divide the loss by batch_size
        loss = self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1))
        self.log("test_loss", loss)

        return {"batch_index": batch_index, "val_loss":loss}
    

