import argparse
import os

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import GPTMatchingModel
from metrics import MatchingMetrics
from utils import CheckpointManager
from minGPT.mingpt.bpe import BPETokenizer

def evaluate_jd_cv_pairs(model, data, tokenizer, block_size, device, metrics_calculator, calc_loss=False, verbose=True):
    """Evaluate model on JD-CV pairs by averaging chunk scores"""
    model.eval()
    
    all_scores = []
    all_labels = []
    val_loss = 0 if calc_loss else None
    norm_loss = 0 if calc_loss else None
    
    # Process each JD-CV pair
    iter = 0
    for _, row in data.iterrows():
        jd_text = row['job_description_text']
        cv_text = row['resume_text']
        label = row['label']
        
        # Tokenize full texts
        jd_tokens_list = tokenizer(jd_text).squeeze(0).tolist()
        cv_tokens_list = tokenizer(cv_text).squeeze(0).tolist()
        
        # Create chunks with overlap
        jd_chunks = []
        cv_chunks = []
        
        # Parameters for chunking
        assert block_size % 2 == 0, "Block size must be even"
        jd_size = block_size // 2 
        cv_size = block_size // 2 - 1 # Leave space for separator token
        overlap = 128
        # token ID of separator 
        sep_token_id = tokenizer('###').item()
        
        # Create JD chunks on CPU
        for i in range(0, len(jd_tokens_list), jd_size - overlap):
            chunk = jd_tokens_list[i:i + jd_size]
            jd_chunks.append(chunk)
        
        # Create CV chunks on CPU
        for i in range(0, len(cv_tokens_list), cv_size - overlap):
            chunk = cv_tokens_list[i:i + cv_size]
            cv_chunks.append(chunk)
        
        # Create all combinations of chunks
        chunk_scores = []
        for jd_chunk in jd_chunks:
            for cv_chunk in cv_chunks:
                # Combine chunks with separator
                combined = torch.tensor(jd_chunk + [sep_token_id]  + cv_chunk)
               
                # Process batch
                combined = combined.unsqueeze(0).to(device)  # Add batch dimension
                target = torch.tensor([metrics_calculator.label_map[label]]).to(device)
                with torch.no_grad():
                    score, loss = model(combined, target)
                    if calc_loss:
                        val_loss += loss.item()
                        norm_loss += 1
                chunk_scores.append(score.item())
        
        # Average scores for this JD-CV pair
        pair_score = np.mean(chunk_scores)
        all_scores.append(pair_score)
        all_labels.append(float(metrics_calculator.label_map[label]))
        
        if verbose and (iter + 1) % 10 == 0:
            print(f"Processed {iter + 1}/{len(data)} pairs")
        
        iter += 1
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_metrics(
        np.array(all_scores),
        np.array(all_labels)
    )

    loss = val_loss/norm_loss if calc_loss else None
    
    return metrics, all_scores, all_labels, loss

def plot_confusion_matrix(conf_matrix, save_path):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fit', 'Potential Fit', 'Good Fit'],
                yticklabels=['No Fit', 'Potential Fit', 'Good Fit'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate JD-Resume matching model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--model_name', type=str, default='gpt2_b1024_l12_h12_e768_lora_nowd_lt0.3_ht0.8')
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    parser.add_argument('--test_data', type=str, default='data/test.csv')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    
    # Load model
    model_path = os.path.join(args.checkpoint_dir, args.model_name)
    checkpoint_path = os.path.join(model_path, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model_config.model_type = None
    model = GPTMatchingModel(model_config)
    checkpoint_manager = CheckpointManager(model_path, model_config=model_config)
    checkpoint = checkpoint_manager.load_checkpoint(model, checkpoint_path=checkpoint_path, map_location=device)
    model = model.to(device)

    tokenizer = BPETokenizer()
    
    # for v0 models
    try:
        thresholds = model.config.thresholds
        label_map=model.config.label_map
    except AttributeError:
        thresholds = [0.4, 0.75]
        label_map={
                'No Fit': 0.0,
                'Potential Fit': 0.6,
                'Good Fit': 1.0
            }
        
    # Initialize metrics calculator
    metrics_calculator = MatchingMetrics(label_map, thresholds)

    # Load original data
    data = pd.read_csv(args.test_data)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, scores, labels, _ = evaluate_jd_cv_pairs(
        model=model,
        data=data,
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        device=device,
        metrics_calculator=metrics_calculator
    )
    
    # Print and save results
    print("\nEvaluation Results:")
    metrics_calculator.print_metrics(metrics)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    
    # Save metrics
    with open(results_path, 'w') as f:
        f.write(f"Model: {args.model_name}\n\n")
        f.write("Evaluation Results:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n\n")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        
        f.write("Per-Class Metrics:\n")
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"\n{class_name}:\n")
            for metric_name, value in class_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
    
    # Save predictions
    data['predicted_score'] = scores
    data['true_label'] = labels
    data.to_csv(predictions_path, index=False)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()