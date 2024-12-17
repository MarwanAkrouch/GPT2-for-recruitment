import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from typing import List

class JDResumeDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame ,
        tokenizer,
        block_size: int = 1024,
        num_tokens_jd: int = 512,
        num_tokens_cv: int = 511,
        overlap_jd: int = 128,
        overlap_cv: int = 128,
        label_map={
            'No Fit': 0.0,
            'Potential Fit': 0.5,
            'Good Fit': 1.0
        },
        
    ):
        self.data = data_df 
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.num_tokens_jd = num_tokens_jd
        self.num_tokens_cv = num_tokens_cv
        self.overlap_jd = overlap_jd
        self.overlap_cv = overlap_cv
        
        # Label mapping
        self.label_map = label_map

        # Chunk generation multipliers to balance the dataset
        self.chunk_multipliers = {
            'No Fit': 2,      # Lowest number of chunks
            'Potential Fit': 3,  # More chunks
            'Good Fit': 3     # More chunks
        }

        self.sep_token_id = self.tokenizer('###').item() # token ID of separator token
        self.pad_token_id = 0 # 50256

        # Verify the configuration
        assert num_tokens_jd + num_tokens_cv + 1 <= block_size, "Combined chunk sizes plus separator exceed block size"
        assert overlap_jd < num_tokens_jd, "JD overlap must be smaller than chunk size"
        assert overlap_cv < num_tokens_cv, "CV overlap must be smaller than chunk size"
        
        # Process all texts and create chunk pairs
        self.chunk_pairs = []
        self.chunk_labels = []
        
        # Process each row in the dataframe
        for _, row in self.data.iterrows():
            jd = row['job_description_text']
            cv = row['resume_text']
            label = row['label']
            label_value = self.label_map[label]
            
            # Generate chunks based on label multiplier
            multiplier = self.chunk_multipliers[label]
            
            # Create chunk pairs with adjusted multiplier
            pairs = self._create_chunk_pairs(jd, cv, multiplier)
            
            self.chunk_pairs.extend(pairs)
            self.chunk_labels.extend([label_value] * len(pairs))
        
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        return self.tokenizer(text, return_tensors='pt').squeeze(0).tolist()
    
    def _create_chunks(self, tokens: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
        """Create overlapping chunks from tokens."""
        chunks = []
        stride = chunk_size - overlap
        
        # Handle case where text is shorter than chunk size
        if len(tokens) <= chunk_size:
            return [tokens + [self.pad_token_id] * (chunk_size - len(tokens))]  # Using 0 as pad token
        
        # Create overlapping chunks
        for i in range(0, len(tokens) - overlap, stride):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk if needed
                chunk = chunk + [self.pad_token_id] * (chunk_size - len(chunk))
            chunks.append(chunk)
        
        return chunks
    
    
    def _create_chunk_pairs(self, jd: str, cv: str, multiplier: int = 1) -> List[List[int]]:
        """Create chunk pairs with multiplier for oversampling"""
        # Tokenize texts
        jd_tokens = self._tokenize_text(jd)
        cv_tokens = self._tokenize_text(cv)
        
        # Create chunks
        jd_chunks = self._create_chunks(jd_tokens, self.num_tokens_jd, self.overlap_jd)
        cv_chunks = self._create_chunks(cv_tokens, self.num_tokens_cv, self.overlap_cv)
        
        # Create all possible pairs with multiplier
        pairs = []
        for _ in range(multiplier):
            for jd_chunk in jd_chunks:
                for cv_chunk in cv_chunks:
                    # Combine chunks with separator
                    combined = (
                        jd_chunk +
                        [self.sep_token_id] +
                        cv_chunk
                    )
                    pairs.append(combined)
        
        return pairs
    
    def print_info(self):
        """Print comprehensive dataset information"""
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        # Basic stats
        print("\n1. Basic Statistics:")
        print(f"Total samples: {len(self.data)}")
        print(f"Total chunks: {len(self.chunk_pairs)}")
        print(f"Average chunks per sample: {len(self.chunk_pairs)/len(self.data):.1f}")
        
        # Label distribution
        print("\n2. Label Distribution:")
        label_counts = self.data['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count/len(self.data)) * 100
            print(f"{label}: {count} samples ({percentage:.1f}%)")
        
        # Text length statistics
        print("\n3. Text Length Statistics:")
        
        print("\nJob Descriptions (characters):")
        jd_lengths = self.data['job_description_text'].str.len()
        print(f"Mean length: {jd_lengths.mean():.0f}")
        print(f"Median length: {jd_lengths.median():.0f}")
        print(f"Min length: {jd_lengths.min()}")
        print(f"Max length: {jd_lengths.max()}")
        
        print("\nResumes (characters):")
        cv_lengths = self.data['resume_text'].str.len()
        print(f"Mean length: {cv_lengths.mean():.0f}")
        print(f"Median length: {cv_lengths.median():.0f}")
        print(f"Min length: {cv_lengths.min()}")
        print(f"Max length: {cv_lengths.max()}")
        
        # Token statistics
        print("\n4. Token Statistics:")
        print(f"Block size: {self.block_size}")
        print(f"JD tokens per chunk: {self.num_tokens_jd}")
        print(f"CV tokens per chunk: {self.num_tokens_cv}")
        print(f"JD overlap: {self.overlap_jd}")
        print(f"CV overlap: {self.overlap_cv}")
        
        # Sample tokenization
        print("\n5. Tokenization Example:")
        sample_idx = np.random.randint(len(self.data))
        sample_jd = self.data['job_description_text'].iloc[sample_idx][:500]
        sample_cv = self.data['resume_text'].iloc[sample_idx][:500]
        
        jd_tokens = len(self._tokenize_text(sample_jd))
        cv_tokens = len(self._tokenize_text(sample_cv))
        
        print("Random sample (first 500 chars):")
        print(f"JD tokens: {jd_tokens}")
        print(f"CV tokens: {cv_tokens}")
        
        # Chunk distribution
        print("\n6. Chunk Distribution by Label:")
        chunk_labels = np.array(self.chunk_labels)
        for label_name, label_value in self.label_map.items():
            n_chunks = np.sum(chunk_labels == label_value)
            print(f"{label_name}: {n_chunks} chunks ({n_chunks/len(self.chunk_pairs)*100:.1f}%)")
        
        print("\n" + "="*50 + "\n")

    
    def __len__(self):
        return len(self.chunk_pairs)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.chunk_pairs[idx], dtype=torch.long)
        label = torch.tensor(self.chunk_labels[idx], dtype=torch.float)
        
        return tokens, label

def load_data_train_val_split(data_path: str, val_ratio: float = 0.1):
    """Load data from CSV file and split into training and validation sets."""
    
    data = pd.read_csv(data_path)
    train_df, val_df = train_test_split(data, test_size=val_ratio, random_state=42)

    # Print new validation distribution
    print("\nValidation distribution after stratification:")
    print(val_df['label'].value_counts(normalize=True))
    
    return train_df, val_df
