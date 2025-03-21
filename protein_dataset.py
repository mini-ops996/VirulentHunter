from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer: AutoTokenizer, max_length=1000):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx][:self.max_length]
        label    = self.labels[idx]
        encoding = self.tokenizer(sequence, truncation=True, 
                                  padding='max_length', max_length=self.max_length, 
                                  return_tensors='pt')
        encoding['label'] = label

        return encoding