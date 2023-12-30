from typing import Any, Tuple, List
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import lightning as L

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, d_model, device=None, dtype: torch.dtype=torch.float32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dk = torch.sqrt(torch.scalar_tensor(d_model, device=device, dtype=dtype))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input should have (B,N,d_model)
        # q (b,1,d_model) , k (b,n,d_model)
        # qk = (b,1,n)
        scaled_qk = q@torch.transpose(k, 2, 1) / self.dk
        if mask is not None:
            masked_scaled_qk = torch.masked_fill(scaled_qk, mask=mask.bitwise_not(), value=-1e4)
        attention_weights = torch.softmax(masked_scaled_qk, dim=-1)
        return  attention_weights @  v
        

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.depth = d_model // n_head

        self.q_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))
        self.k_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))
        self.v_linear = torch.nn.Sequential(torch.nn.Linear(d_model, d_model, device=device, dtype=dtype), torch.nn.Dropout(dropout))

        self.attns = torch.nn.ModuleList([ScaledDotProductAttention(d_model=self.depth, device=device, dtype=dtype) for _ in range(n_head)])
        self.output_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.dropout = torch.nn.Dropout(dropout)
                
        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        b,n,_ = input.shape
        q = self.q_linear.forward(input).view((b, n, self.n_head, -1))
        k = self.k_linear.forward(input).view((b, n, self.n_head, -1))
        v = self.v_linear.forward(input).view((b, n, self.n_head, -1))

        attn_output = torch.concat([self.attns[i].forward(q[:,:,i,:].view((b,n,self.depth)), 
                                            k[:,:,i,:].view((b,n, self.depth)), 
                                            v[:,:,i,:].view((b,n, self.depth)),mask=mask) for i in range(self.n_head)],dim=-1)
        return self.dropout.forward(self.output_linear.forward(attn_output))
        


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
        self.mha = MultiHeadAttention(d_model=d_model, n_head=n_head, device=device, dtype=dtype, dropout=dropout)
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
                 d_model:int, n_head:int, num_layers:int, pad_id:int=None,  device=None, 
                 dtype:torch.dtype=torch.float32, dropout:float=0.2, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['dtype', 'device'])
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.decay = weight_decay
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_id, device=device, dtype=dtype)
        self.transformers = torch.nn.Sequential(*[Transformer(n_head=n_head, d_model=d_model, device=device, dtype=dtype, dropout=dropout) for _ in range(num_layers)])
        self.output_linear = torch.nn.Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.decay)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=self.eps, total_iters=2000, end_factor=1),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=2000, eta_min=self.eps)
        ],milestones=[2000])
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "name": "CosineWithWarmUp",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X should have shape of (B,N)
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        return self.output_linear.forward(self.transformers.forward((self.embedding.forward(input=X), None)))
    

    def training_step(self, data: Tuple[torch.Tensor], batch_index:Any, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
  
        X, padding_mask = data["input_ids"], data["attention_mask"]

        input:torch.Tensor = X[:,:-1].long()
        target:torch.Tensor = X[:,1:].long()
        B,n = input.shape

        attention_mask:torch.Tensor = torch.tril(torch.ones((n,n), device=input.device)).unsqueeze(0).expand((B,n,n)) * padding_mask[:,1:].unsqueeze(1)
        attention_mask = attention_mask.bool()

        X_wemb = self.embedding(input)  # dense word vector
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        loss = self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1)) # I think this will get loss for all next token prediction, I mean data[0:-1] vs data[1:0]
        self.log("train_loss", loss)

        return {"batch_index": batch_index, "loss":loss}
    

    def validation_step(self, data: Tuple[torch.Tensor], batch_index,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        # if len(data) < 3:
        #     raise ValueError('data should be tuple of (input, output, attention_mask)')
        # X, y, attention_mask = data # here X, y are both batched array of token_ids (B,N)
        X, padding_mask = data["input_ids"], data["attention_mask"]

        input:torch.Tensor = X[:,:-1]
        target:torch.Tensor = X[:,1:].long()
        B,n = input.shape

        attention_mask:torch.Tensor = torch.tril(torch.ones((n,n), device=input.device)).unsqueeze(0).expand((B,n,n)) * padding_mask[:,1:].unsqueeze(1)
        attention_mask = attention_mask.bool()

        X_wemb = self.embedding(input)  # dense word vector
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        loss = self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1)) # I think this will get loss for all next token prediction, I mean data[0:-1] vs data[1:0]
        self.log("val_loss", loss)

        return {"batch_index": batch_index, "val_loss":loss}
    
    
    def test_step(self, data: Tuple[torch.Tensor], batch_index, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        # if len(data) < 3:
        #     raise ValueError('data should be tuple of (input, output, attention_mask)')

        X, attention_mask = data["input_ids"], data["attention_mask"]
        input = X[:,:-1,:]
        target = X[:,1:,:]

        
        # X, y, attention_mask = data # here X, y are both batched array of token_ids (B,N)
        
        X_wemb = self.embedding.forward(input)  # dense word vector
        hidden_output, _ = self.transformers.forward((X_wemb, attention_mask))

        logits = self.output_linear.forward(hidden_output)
        loss = self.loss(logits.view(-1, logits.size(-1)), target.view(-1)) # I think this will get loss for all next token prediction, I mean data[0:-1] vs data[1:0]
        return {"batch_index": batch_index, "test_loss":loss}
    

