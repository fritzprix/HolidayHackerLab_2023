from typing import Any, Tuple, List
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning as L
class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, d_model, device=None, dtype: torch.dtype=torch.float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dk = torch.sqrt(torch.scalar_tensor(d_model, device=device, dtype=dtype))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input should have (B,N,d_model)
        # q (b,1,d_model) , k (b,n,d_model)
        # qk = (b,1,n)
        scaled_qk = q@torch.transpose(k, 2, 1) / self.dk
        if mask is not None:
            scaled_qk = scaled_qk * mask
        attention_weights = torch.softmax(scaled_qk, dim=-1)
        return  attention_weights @  v
        

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.depth = d_model // n_head

        self.q_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)

        self.attns = torch.nn.ModuleList([ScaledDotProductAttention(d_model=self.depth, device=device, dtype=dtype) for _ in range(n_head)])
        self.output_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
                
        
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
        return self.output_linear.forward(attn_output)
        


class PositionWiseFeedforward(torch.nn.Module):

    def __init__(self, d_model:int, device, dtype: torch.dtype=torch.float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pwff = torch.nn.Sequential(torch.nn.Linear(d_model, d_model * 4, device=device,dtype=dtype), 
                                            torch.nn.GELU(), 
                                            torch.nn.Linear(4* d_model, d_model, device=device, dtype=dtype))
        
    def forward(self, input: torch.Tensor)-> torch.Tensor:
        return self.pwff.forward(input)


class Transformer(torch.nn.Module):

    def __init__(self, n_head, d_model, device, dtype:torch.dtype=torch.float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_norm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(d_model=d_model, n_head=n_head, device=device, dtype=dtype)
        self.mha_lnorm = torch.nn.LayerNorm(d_model, device=device,dtype=dtype)
        self.pw_ff = PositionWiseFeedforward(d_model=d_model, device=device, dtype=dtype)

    def forward(self, input:torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # Pre-LayerNormalization from GPT-3, (note: Post-LayerNormalization is used for GPT-2 and original paper)
        norm_input = self.input_norm.forward(input)
        mha_output = input + self.mha.forward(norm_input, mask)
        norm_mha_output = self.mha_lnorm(mha_output)
        return mha_output + self.pw_ff.forward(norm_mha_output)
    

class ToyGPT(L.LightningModule):

    def __init__(self, 
                 vocab_size:int, 
                 d_model:int, n_head:int, num_layers:int, pad_id:int=None,  device=None, dtype:torch.dtype=torch.float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_id, device=device, dtype=dtype)
        self.transformers = torch.nn.Sequential(*[Transformer(n_head=n_head, d_model=d_model, device=device, dtype=dtype) for _ in range(num_layers)])
        self.output_linear = torch.nn.Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X should have shape of (B,N)
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        return self.output_linear.forward(self.transformers.forward(self.embedding.forward(input=X)))
    

    def training_step(self, data: torch.Tensor, batch_index:Any, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

        if len(data) < 3:
            raise ValueError('data should be tuple of (input, output, attention_mask)')
        X, y, attention_mask = data # here X, y are both batched array of token_ids (B,N)
        
        if X.shape != y.shape: # X,y shoud have B,N
            raise ValueError('X and y should have same shape')
        
        X_wemb = self.embedding.forward(X)  # dense word vector
        hidden_output = self.transformers.forward(X_wemb, attention_mask)
        logits = self.output_linear.forward(hidden_output)

        loss = self.loss(logits, y)
        return {"train_loss":loss}
    
    def test_step(self, data, batch_index, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

        if len(data) < 3:
            raise ValueError('data should be tuple of (input, output, attention_mask)')
        X, y, attention_mask = data

        if X.shape != y.shape:
            raise ValueError('X and y should have same shape')
        
        X_wemb = self.embedding.forward(X)
        hidden_output = self.transformers.forward(X_wemb, attention_mask)
        logits = self.output_linear.forward(hidden_output)

        loss = self.loss(logits, y)
        return {"test_loss":loss}
    

