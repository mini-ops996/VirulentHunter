import argparse
import os
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from Bio import SeqIO
import torch
import numpy as np
import gc

def main():                                     
    parser = argparse.ArgumentParser(prog='PLM_VF for Virulence Factor Predict.')
    parser.add_argument('--base_esm_path', default='/mnt/data/cs/ESM2_Model/esm2_t30_150M_UR50D', 
                        help='The path of esm2 model.') 
    

    parser.add_argument('--ft_model_path', default='models/binary/',
                        help='the fine turn model')
    parser.add_argument('--catogery_model_path', default='models/category/')
    parser.add_argument('-i','--input_fasta_path', default='data/test.fasta', 
                        help='the input fasta and label file path.')
    parser.add_argument('-o','--output_path', default='results/', help='the predict results to save')
    parser.add_argument('--max_len', default=1000, help='protein sequence length')

    args = parser.parse_args()
    print(args)
    predict(args.base_esm_path, args.ft_model_path, args.catogery_model_path,
            args.input_fasta_path, args.output_path ,args.max_len)

def predict(base_esm_path, ft_model_path, catogery_model_path,
            input_fasta_path, output_path,max_length=1000):
    
    print(f'Load ESM2 model and fine tune model.')
    tokenizer = AutoTokenizer.from_pretrained(base_esm_path)
    base_esm_model = AutoModelForSequenceClassification.from_pretrained(base_esm_path)
    print(f'Load Fine Tune model.')
    binary_model       = PeftModel.from_pretrained(base_esm_model, ft_model_path)
    binary_model.eval()
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  
    binary_model.to(device)

    label_info = pd.read_csv('data/labels.csv')

    print(f'Read fasta from {input_fasta_path}')
    sequences_dict = {}
    for record in SeqIO.parse(input_fasta_path, "fasta"):
        sequences_dict[record.id] = record.seq
    
    # binary_model
    binary_logits = {}     
    for seq_id, sequence in tqdm(sequences_dict.items()):       
        encoding = tokenizer(str(sequence), truncation=True, return_tensors='pt', 
                                  padding='max_length', max_length=max_length)
        encoding = encoding.to(device)
        with torch.no_grad():   # 
            outputs = binary_model(**encoding) # 
            logits = outputs.logits        # 
            binary_logits[seq_id] = np.round(torch.nn.functional.softmax(logits,dim=-1).cpu().tolist()[0],3)

    del binary_model, base_esm_model
    gc.collect()

    prob_df = pd.DataFrame(binary_logits.values(), binary_logits.keys(), columns=['no_vf_prob', 'vf_prob']) 
    prob_df['id'] = prob_df.index
    prob_df = prob_df[['id', 'vf_prob']]
    prob_df.reset_index(drop=True, inplace=True)

    for cat in label_info['category'].unique():
        prob_df[cat] = 0.0

    # category_model
    base_esm_model = AutoModelForSequenceClassification.from_pretrained(base_esm_path,
                                                                        num_labels=14,)
    category_model     = PeftModel.from_pretrained(base_esm_model, catogery_model_path)
    category_model.eval()
    category_model.to(device)

    for seq_id, sequence in tqdm(sequences_dict.items()):
        vf_prob = prob_df.loc[prob_df['id']==seq_id, 'vf_prob'].values[0]
        if vf_prob >=  0.5:
            encoding = tokenizer(str(sequence), truncation=True, return_tensors='pt', 
                                  padding='max_length', max_length=max_length)
            encoding = encoding.to(device)
            with torch.no_grad():   
                outputs = category_model(**encoding) 
                logits = outputs.logits        
                probs = torch.nn.functional.sigmoid(logits)
                probs = np.round(probs.cpu().numpy().squeeze().tolist(),3).tolist()
                prob_df.loc[prob_df['id']==seq_id, 'Exotoxin':'Regulation'] = probs

    
    prob_df.to_csv(os.path.join(output_path, f'predict_results.csv'), sep=',')

    del base_esm_model,category_model, tokenizer
    gc.collect()


if __name__ == '__main__':
    main()