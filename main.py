import sys, os
from datetime import datetime

from minGPT.mingpt.bpe import BPETokenizer
from minGPT.mingpt.utils import set_seed, setup_logging, CfgNode as CN

import torch

from trainer import MatchingTrainer
from data import JDResumeDataset, load_data_train_val_split
from model import GPTMatchingModel

# -----------------------------------------------------------------------------

def get_config():

    C = CN()
    C.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/edwin/'

    # model
    C.model_type = 'gpt2'

    # map from classification labels to model labels
    C.label_map={
            'No Fit': 0.0,
            'Potential Fit': 0.6,
            'Good Fit': 1.0
        }
    C.thresholds = [0.4, 0.75] # thresholds for the way back to classification

    # trainer\
    C.val_ratio = 0.1
    C.batch_size = 8
    C.num_workers = 4
    C.learning_rate = 1e-4
    C.max_iters = 50000
    C.eval_interval = 200
    C.gradient_accumulation_steps = 8
    C.weight_decay = 1e-6
    C.use_lora = False

    return C

model_configurations = {
    'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024),  # 117M params
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, block_size=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, block_size=1024), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, block_size=1024), # 1558M params
    'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512, vocab_size=50257, block_size=1024),
    'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192, vocab_size=50257, block_size=1024),
    'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128, vocab_size=50257, block_size=1024),
    'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48, vocab_size=50257, block_size=1024),
}
    
if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)
     
    tokenizer = BPETokenizer()

    model = GPTMatchingModel.from_pretrained(config.model_type)
    model.config.use_lora = config.use_lora
    model.config.label_map = config.label_map
    model.config.thresholds = config.thresholds

    train_df, val_df = load_data_train_val_split('data/train.csv', val_ratio=config.val_ratio)

    train_dataset = JDResumeDataset(
        data_df=train_df,
        tokenizer=tokenizer,
        block_size=model_configurations[config.model_type]['block_size'],
        num_tokens_jd=512,
        num_tokens_cv=511,
        overlap_jd=128,
        overlap_cv=128,
        label_map=config.label_map
    )

    train_dataset.print_info()

    # Create config
    trainer_config = MatchingTrainer.get_default_config()
    trainer_config.batch_size = config.batch_size
    trainer_config.max_iters = config.max_iters
    trainer_config.eval_interval = config.eval_interval
    trainer_config.learning_rate = config.learning_rate
    trainer_config.num_workers = config.num_workers
    trainer_config.device = config.device
    trainer_config.gradient_accumulation_steps = config.gradient_accumulation_steps
    trainer_config.weight_decay = config.weight_decay
    

    # Create trainer
    trainer = MatchingTrainer(
        config=trainer_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_df=val_df,
        #additional_info=f"nowd_lt{config.thresholds[0]}_ht{config.thresholds[1]}"
    )

    
    t0 = datetime.now()
    
    # Train
    trainer.run()

    elapsed_time = datetime.now() - t0
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training time: {int(hours)}h {int(minutes)}min {int(seconds)}s")
