import torch
import numpy as np

def predict_single_pair(model, jd_text, cv_text, tokenizer, block_size, device):
    """Get prediction score for a single JD-CV pair"""
    
    # to be able to use v0 models
    try:
        sep_token = model.config.sep_token
        pad_token_id = model.config.pad_token_id
    except AttributeError:
        sep_token = '###'
        pad_token_id = 0

    with torch.device(device):
        model.eval()
        
        # Tokenize full texts
        jd_tokens_list = tokenizer(jd_text).squeeze(0).tolist()
        cv_tokens_list = tokenizer(cv_text).squeeze(0).tolist()
        
        jd_chunks = []
        cv_chunks = []
        
        jd_size = block_size // 2 
        cv_size = block_size // 2 - 1
        overlap = 128
        
        # Create JD chunks on CPU
        for i in range(0, len(jd_tokens_list), jd_size - overlap):
            chunk = jd_tokens_list[i:i + jd_size]
            jd_chunks.append(chunk)
        
        # Create CV chunks on CPU
        for i in range(0, len(cv_tokens_list), cv_size - overlap):
            chunk = cv_tokens_list[i:i + cv_size]
            cv_chunks.append(chunk)
        
        # Get scores on CPU
        chunk_scores = []
        with torch.no_grad():
            for jd_chunk in jd_chunks:
                for cv_chunk in cv_chunks:
                    combined = torch.tensor(jd_chunk + [tokenizer(sep_token).item()]  + cv_chunk)

                    # Pad or trim combined sequence to block_size
                    if len(combined) < block_size:
                        padding = torch.full((block_size - len(combined),), pad_token_id, dtype=torch.long)
                        combined = torch.cat([combined, padding])
                
                    combined = combined.unsqueeze(0)
                    score, _ = model(combined)
                    chunk_scores.append(score.item())
    
    return np.mean(chunk_scores)