
import logging
import sys
import torch
def setup_logger(log_dir: str):
    """配置全局日志记录器"""
    log_filename = f"{log_dir}/training.log"
    
    # 创建logger
    logger = logging.getLogger("QwenFinetune")
    logger.setLevel(logging.INFO)

    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # 创建文件handler
    fh = logging.FileHandler(log_filename, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # 添加handlers到logger
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    打印模型中可训练参数的数量和比例。
    这在 LoRA 微调中非常有用，可以确认只有适配器参数是可训练的。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # 使用 logging 来输出，而不是 print，以便记录到日志文件中
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )