import os
import numpy as np
from Bio import SeqIO
import pandas as pd
from sklearn.metrics import(
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import random

from sty import fg, bg
from sty import Style, RgbBg

dict_of_amino_acid = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

def parse_csv_input(input_path):
    df = pd.read_csv(input_path, index_col=0)
    sequences = list(df['Sequence'])
    labels = list(df['label'])
    return sequences, labels

def parse_inputs(input_fasta_path, input_label_path):
    
    print(f'Read fasta and label file from {input_fasta_path}')
    train_fasta_list = list(SeqIO.parse(os.path.join(input_fasta_path,'train.fasta'), 'fasta'))
    train_labels_df  = pd.read_csv(os.path.join(input_label_path, 'train_labels.csv'), index_col=0)

    assert len(train_fasta_list) == len(train_labels_df), "length of fasta and labels not equal"

    train_sequences = [str(record.seq) for record in train_fasta_list]
    train_labels = [train_labels_df[train_labels_df['id'] == record.id]['label'].values[0] for record in train_fasta_list]
    
    val_fasta_list = list(SeqIO.parse(os.path.join(input_fasta_path,'val.fasta'), 'fasta'))
    val_labels_df  = pd.read_csv(os.path.join(input_label_path, 'val_labels.csv'), index_col=0)

    assert len(val_fasta_list) == len(val_labels_df), "length of fasta and labels not equal"

    val_sequences = [str(record.seq) for record in val_fasta_list]
    val_labels = [val_labels_df[val_labels_df['id'] == record.id]['label'].values[0] for record in val_fasta_list]

    random.seed(42)
    train_indices = list(range(len(train_sequences)))
    random.shuffle(train_indices)
    train_sequences = [train_sequences[i] for i in train_indices]
    train_labels    = [train_labels[i] for i in train_indices]

    val_indices = list(range(len(val_sequences)))
    random.shuffle(val_indices)
    val_sequences = [val_sequences[i] for i in val_indices]
    val_labels    = [val_labels[i] for i in val_indices]

    return train_sequences, train_labels, val_sequences, val_labels

def parse_inputs_multi_label(input_fasta_path, input_label_path):
    print(f'Read fasta, label and mapping file from {input_fasta_path}')
    fasta_list = list(SeqIO.parse(input_fasta_path, 'fasta'))

    with open(input_label_path, 'rb') as f:
        id2lables_dict = pickle.load(f)
    assert len(fasta_list) == len(id2lables_dict), "length of fasta and labels not equal"

    sequences = [str(record.seq) for record in fasta_list]
    if isinstance(id2lables_dict, pd.DataFrame):
        labels = list(id2lables_dict['label'])
    else:
        labels = [id2lables_dict[record.id] for record in fasta_list]

    return sequences, labels


def parse_inputs_category(input_fasta_path, input_label_path):#, input_mapping_path):

    print(f'Read fasta, label and mapping file from {input_fasta_path}')
    fasta_list = list(SeqIO.parse(input_fasta_path, 'fasta'))
    labels_df  = pd.read_csv(input_label_path, index_col=0)
    
    
    assert len(fasta_list) == len(labels_df), "length of fasta and labels not equal"
    
    sequences = [str(record.seq) for record in fasta_list]
    labels = [labels_df[labels_df['id'] == record.id]['label'].values[0] for record in fasta_list]

    return sequences, labels

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    # compute accuracy
    accuracy = accuracy_score(labels, predictions)

    # compute precision, recall, F1 score, AUC
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)

    # compute MCC
    mcc = matthews_corrcoef(labels, predictions)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1':f1,
            'auc': auc, 'mcc': mcc}

def compute_multi_label_metrics(p):
    predictions, labels = p
    
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_weighted': f1_weighted_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,):
    '''
    comes from https://github.com/DTrimarchi10/confusion_matrix
    '''
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[1,:])
            recall    = cf[1,1] / sum(cf[:,1])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


def normalize_for_color(force_of_highlights):
    # Min max normalization
    force_of_highlights = (force_of_highlights - np.min(force_of_highlights)) / (
        np.max(force_of_highlights) - np.min(force_of_highlights)
    )
    force_of_highlights = force_of_highlights * 255
    force_of_highlights = force_of_highlights.astype(int)

    return force_of_highlights

def get_color_for_viz(force_of_highlights, method):
    force_of_highlights = np.array(force_of_highlights)
    if method == "classic":
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "classic_with_saturation":
        force_of_highlights = normalize_for_color(force_of_highlights)
        # Have more than 255 values
        force_of_highlights = force_of_highlights * 2
        # Clip value to be correct colors
        force_of_highlights = np.clip(force_of_highlights, 0, 255)
    elif method == "binary_with_threashold":
        # Discrete version with the optimal threashold
        force_of_highlights[force_of_highlights >= 0.11828187716390] = 1
        force_of_highlights[force_of_highlights < 0.11828187716390] = 0
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "rank_colorize":
        force_of_highlights = np.argsort(force_of_highlights)
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "first_percentile_colorize":
        # Les 5% des plus fort highlights en découpant en tranche de 0.5 ?
        min_percentile = 95
        step = 0.05  # 0.5
        nb_step = int((100 - min_percentile) / step)
        list_percentile = [min_percentile + k * step for k in range(nb_step)]
        new_force_of_highlights = np.zeros(force_of_highlights.shape)
        # print("list_percentile :",list_percentile)
        for ind, percentile in enumerate(list_percentile): # 100等份
            threashold = np.percentile(force_of_highlights, percentile)
            new_force_of_highlights[force_of_highlights > threashold] = ind
        force_of_highlights = normalize_for_color(new_force_of_highlights)
    return force_of_highlights


def print_color_text(
    sequence, intensite_color, method="first_percentile_colorize", composante="green"
):
    force_of_highlights = intensite_color

    force_of_highlights = get_color_for_viz(force_of_highlights, method)

    intensite_color = force_of_highlights

    sequence_highlight = fg.white + ""
    symbolcount = 0
    for letter, c in zip(sequence, intensite_color):
        c = int(c)

        if composante == "green":
            bg.new_color = Style(RgbBg(255 - c, 255, 255 - c))
        elif composante == "red":
            bg.new_color = Style(RgbBg(255, 255 - c, 255 - c))
        elif composante == "blue":
            bg.new_color = Style(RgbBg(255 - c, 255 - c, 255))

        if symbolcount % 10 == 0:
            sequence_highlight += bg(255, 255, 255) + " "
        if symbolcount % 100 == 0:
            sequence_highlight += (
                "\n\n" + bg(255, 255, 255) + str(int(symbolcount / 100)) + " "
            )

        sequence_highlight += bg.new_color + letter
        symbolcount += 1

    print(sequence_highlight)


def unit_length_norm2_normalize(vec_score):
    vec_score = np.array(vec_score, dtype=np.float64)
    norm_vec_score = np.sqrt(np.sum(np.power(vec_score, 2)))
    norm_vec = np.divide(
        vec_score,
        norm_vec_score,
        out=np.zeros_like(vec_score),
        where=norm_vec_score != 0,
    )
    return norm_vec

def convert_number_to_colors(scores, ):
    all_couleur = []
    for red_comp in scores:
        couleur = "%02x%02x%02x" % (255, 255 - red_comp, 255 - red_comp)
        couleur = couleur.upper()
        couleur = "0x" + couleur
        all_couleur.append(couleur)
    return all_couleur


import logging
def get_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that includes the current date and time
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Example usage
    logger.info("This is an info message.")
    return logger

def read_fasta(data_path: str, sep=" "):
    """
    Reads a FASTA file and returns a list of tuples containing sequences, ids, and labels.
    """
    sequences_with_ids_and_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        components = record.description.split(sep)
        # labels[0] contains the sequence ID, and the rest of the labels are GO terms.
        sequence_id = components[0]
        labels = components[1:]

        # Return a tuple of sequence, sequence_id, and labels
        sequences_with_ids_and_labels.append((sequence, sequence_id, labels))
    return sequences_with_ids_and_labels



AA_list = ('G','A','V','L','I','P','F','Y','W','R'
           'S','T','C','M','N','Q','D','E','K','H')

def AA_replace(seq):
    odd_AAs = set()
    for s in seq:
        if s not in AA_list:
            odd_AAs.add(s)
    for k in odd_AAs:
        seq = seq.replace(k,'X')        
    return seq

from esm import FastaBatchedDataset
from torch.utils.data import DataLoader
def extract(fasta_file, alphabet, model,repr_layers=[32], batch_size=500, max_len = 1024, shuffle=True):
    dataset = FastaBatchedDataset.from_file(fasta_file)
    seq_num = len(dataset.sequence_labels)
    for i in range(seq_num):
        dataset.sequence_strs[i] = AA_replace(dataset.sequence_strs[i])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    data_loader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), 
                             batch_size=batch_size, shuffle=shuffle)
    
    print(f"Read {fasta_file} with {len(dataset)} sequences")
    
    repr_layers = [min(i, model.num_layers) for i in repr_layers]
        
    result = {layer:torch.empty([0, ]) for layer in repr_layers}
    seq_id = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            toks = toks[:, :max_len].to(device)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            seq_id.extend(labels)

            representations = out['representations'].items()
            for layer, t  in representations:
                for i, label in enumerate(labels):
                    tmp = t[i, 1 : len(strs[i]) + 1].mean(0).unsqueeze(0).to(device='cpu')
                    result[layer] =  torch.cat((result[layer], tmp),0)
    result = result[repr_layers[0]].detach().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return seq_id, result