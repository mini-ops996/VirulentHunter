import argparse

from train import train_protein_model

def main(): 
    parser = argparse.ArgumentParser(prog='Fine-Tune EMS2 for Virulence Factor Predict.')
    parser.add_argument('--esm_path', default='/ESM2_Model/esm2_t30_150M_UR50D', 
                        help='the path of esm2 model.')
    parser.add_argument('--input_fasta_path', default='binary/', 
                        help='the input fasta and label file path.')
    parser.add_argument('--input_label_path', default='binary/', 
                        help='the input fasta and label file path.')
    parser.add_argument('--max_len', default=2000, help='protein sequence length')

    args = parser.parse_args()
    print(args)

    train_protein_model(args.esm_path, args.input_fasta_path, args.input_label_path,args.max_len)

if __name__ == '__main__':
    main()