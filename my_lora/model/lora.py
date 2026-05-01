# src/lora.py

import math
import logging
from typing import List

import torch
from torch import nn
import bitsandbytes as bnb

class LoraLayer(nn.Module):
    """
    手动实现的LoRA层，用于包裹一个标准的线性层（或4bit量化层）。
    """
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer

        if isinstance(original_layer, nn.Linear):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        elif isinstance(original_layer, bnb.nn.Linear4bit):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(original_layer)}")

        self.rank = rank
        self.lora_alpha = lora_alpha
        
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.rank

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始的量化层会正确处理输入，其输出为计算精度（如 bfloat16）。
        original_result = self.original_layer(x)

        
        lora_input_dtype = self.lora_A.weight.dtype
        lora_result = self.lora_B(self.lora_A(self.lora_dropout(x.to(lora_input_dtype))))

        # LoRA旁路计算的结果是 float32。我们需要把它转换回
        # 原始输出的精度，才能和它相加。
        return original_result + (lora_result * self.scaling).to(original_result.dtype)

def apply_lora_to_model(
    model: nn.Module, 
    rank: int, 
    lora_alpha: int, 
    lora_dropout: float,
    target_modules: List[str]
) -> nn.Module:
    """
    遍历模型，找到所有目标模块，并用我们手写的LoraLayer替换它们。
    """
    logging.info("Applying custom LoraLayer to the model...")
    
    for module_name, module in model.named_modules():
        if any(target in module_name for target in target_modules):
            if isinstance(module, (nn.Linear, bnb.nn.Linear4bit)):
                parent_module_name = '.'.join(module_name.split('.')[:-1])
                child_module_name = module_name.split('.')[-1]
                parent_module = model.get_submodule(parent_module_name)
                
                device = next(model.parameters()).device
                
                lora_layer = LoraLayer(
                    original_layer=module,
                    rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                ).to(device)
                
                setattr(parent_module, child_module_name, lora_layer)
                logging.info(f"Replaced {module_name} with LoraLayer on device {device}.")
    
    
    if hasattr(model, "enable_input_require_grads"):
        # 某些Hugging Face模型内置了此方法
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            input_embeddings.register_forward_hook(make_inputs_require_grad)
            logging.info("Attached forward hook to input embeddings to enable gradients for checkpointing.")
    # -----------------------------------------

    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False

    return model
