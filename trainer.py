import math
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from minGPT.mingpt.trainer import Trainer
from metrics import MatchingMetrics
from utils import CheckpointManager
from evaluate import evaluate_jd_cv_pairs

class MatchingTrainer(Trainer):
    @staticmethod
    def get_default_config():
        C = Trainer.get_default_config()
        C.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        C.eval_interval = 300
        C.warmup_iters = 100
        C.lr_decay_iters = 5000
        C.min_lr = 5e-6
        C.grad_norm_clip = 1.0
        C.betas = (0.9, 0.95)
        C.eps=1e-8
        C.checkpoint_dir = 'checkpoints'
        C.keep_last_n = 5    # Keep last N checkpoints
        C.gradient_accumulation_steps = 4
        C.weight_decay = .1
        return C

    def __init__(self, config, model, train_dataset, tokenizer, val_df=None, additional_info=None):
        super().__init__(config, model, train_dataset)
        self.val_df = val_df
        self.best_val_metric = 0
        self.total_samples = len(train_dataset)
        self.val_metrics_history = []
        self.metrics_calculator = MatchingMetrics(model.config.label_map, model.config.thresholds)
        self.additional_info = additional_info
        self.tokenizer = tokenizer

        # Initialize checkpoint manager with model config
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            model_config=model.config,  # Pass the model's config
            additional_info=additional_info
        )

        # Initialize tensorboard writer
        model_id = self.checkpoint_manager._get_model_identifier(additional_info)
        log_dir = os.path.join('runs', model_id)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    
    def _get_epoch(self, iter_num):
        samples_processed = iter_num * self.config.batch_size
        return samples_processed / self.total_samples
    
    def _get_lr(self, it):
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def run(self):
        model, config = self.model, self.config
        
        # Setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        
        # Setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Load latest checkpoint if exists
        checkpoint = self.checkpoint_manager.load_checkpoint(model, self.optimizer)
        if checkpoint:
            print(f"Resuming from epoch {checkpoint['epoch']:.2f}")
            self.iter_num = checkpoint['iter_num']
            self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            print(f"Resuming from iteration {self.iter_num}")
        
        model.train()
        self.iter_num = 0
        data_iter = iter(train_loader)
        last_epoch = -1
        
        self.optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        while True:
            iteration_start = time.time()
            current_epoch = self._get_epoch(self.iter_num)
            
            if int(current_epoch) > last_epoch:
                print(f"\nStarting Epoch {int(current_epoch)+1}")
                last_epoch = int(current_epoch)
            
            # Adjust learning rate
            lr = self._get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get the next batch
            try:
                tokens, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                tokens, labels = next(data_iter)
            
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)
            
            scores, self.loss = model(tokens, labels)

            self.loss = self.loss / config.gradient_accumulation_steps
            
            self.loss.backward()

            accumulated_loss += self.loss.item()

            if (self.iter_num + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # tensorboard stuff    
                self.writer.add_scalar('train/loss', accumulated_loss, self.iter_num)
                self.writer.add_scalar('train/learning_rate', lr, self.iter_num)
                iteration_time = time.time() - iteration_start
                print(f"iter {self.iter_num} (epoch {current_epoch:.2f}): loss {accumulated_loss:.4f}, lr {lr:e}, Iteration time: {iteration_time:.4f} seconds")
            
                accumulated_loss = 0.0

            # Validation
            if self.val_df is not None and self.iter_num % config.eval_interval == 0:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():

                    # Calculate classification metrics and regression loss
                    val_metrics, all_scores, all_labels, val_loss = evaluate_jd_cv_pairs(model, self.val_df, self.tokenizer, model.block_size, self.device, self.metrics_calculator, calc_loss=True, verbose=False)

                    print(f"\nValidation at iter {self.iter_num} (epoch {current_epoch:.2f}):")
                    print(f"Validation loss: {val_loss:.4f}")
                    
                    
                    self.metrics_calculator.print_metrics(val_metrics)
                    
                    # Store metrics history
                    self.val_metrics_history.append({
                        'iter': self.iter_num,
                        'epoch': current_epoch,
                        'loss': val_loss,
                        'metrics': val_metrics
                    })
                    
                    # Save best model
                    if val_metrics['f1_weighted'] > self.best_val_metric:
                        self.best_val_metric = val_metrics['f1_weighted']
                        print(f"\nNew best validation weighted f1: {val_metrics['f1_weighted']:.4f} at epoch {current_epoch:.2f}")
                        self.checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=self.optimizer,
                            epoch=current_epoch,
                            iter_num=self.iter_num,
                            best_val_metric=val_metrics['f1_weighted'],
                            metrics=val_metrics,
                            is_best=True
                        )
                        self.trigger_callbacks('on_validation_end')

 
                    self.writer.add_scalar('val/loss', val_loss, self.iter_num)
                    
                    # Log classification metrics
                    self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], self.iter_num)
                    self.writer.add_scalar('val/mse', val_metrics['mse'], self.iter_num)
                    self.writer.add_scalar('val/rmse', val_metrics['rmse'], self.iter_num)
                    self.writer.add_scalar('val/f1_weighted', val_metrics['f1_weighted'], self.iter_num)
                    
                    # Log per-class metrics
                    for class_name, metrics in val_metrics['per_class'].items():
                        for metric_name, value in metrics.items():
                            self.writer.add_scalar(
                                f'val/{class_name}/{metric_name}', 
                                value, 
                                self.iter_num
                            )
                
                model.train()
            
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                print(f"\nTraining completed at epoch {current_epoch:.2f}")
                break