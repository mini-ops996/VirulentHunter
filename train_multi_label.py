import os
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from accelerate import Accelerator
import torch.nn.functional as F
import torch
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,

)
from peft import LoraConfig, get_peft_model, TaskType

from protein_dataset import ProteinDataset
from utils import parse_inputs_multi_label, compute_multi_label_metrics
from focal_loss import FocalLoss


TRAIN_SIZE = 0.9
SEED = 42

def wandb_hp_space(trial):
    return{
        "method": "random",
        "metric": {"name": "f1", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-3},
            "per_device_train_batch_size": {"values": [4]}, 
            "weight_decay" :{"values": [0.0,0.2,0.4,0.5]},            
            "max_grad_norm": {"values": [0.2,0.5,1.0]},                       
         },
    }


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")   
        outputs = model(**inputs)
        logits = outputs.logits 

        #loss_fct = FocalLoss(weights=self.class_weights,
        #                     gamma=2)
        #loss = loss_fct(logits, labels.to(torch.float32))
        
        loss = F.binary_cross_entropy_with_logits(logits,
                                                      labels.to(torch.float32),
                                                      pos_weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss

def train_protein_model(esm_path, input_fasta_path, input_label_path,max_length):

    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    sequences, labels = parse_inputs_multi_label(input_fasta_path, input_label_path)
    num_labels = len(labels[0])
    
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(sequences, labels, 
                                                                                    test_size=1-TRAIN_SIZE, 
                                                                                    random_state=SEED)
    train_dataset = ProteinDataset(train_sequences, train_labels, tokenizer, max_length)
    val_dataset = ProteinDataset(val_sequences, val_labels, tokenizer, max_length)

    accelerator = Accelerator()
    train_dataset, val_dataset = accelerator.prepare(train_dataset, val_dataset)

    # class weights
    #class_weights = compute_class_weight('balanced', 
    #                                     classes=np.unique(train_dataset.dataset.labels), 
    #                                     y=train_dataset.dataset.labels)
    
    labels = np.array(labels, dtype=int)
    class_weights = 1 - labels.sum(axis=0) / labels.sum()
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)
    
    def model_init(trial):
        base_model = EsmForSequenceClassification.from_pretrained(
            esm_path, num_labels=num_labels, hidden_dropout_prob=0.2,
            problem_type = "single_label_classification",
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,    
            lora_alpha=32,      
            target_modules=["query", "key", "value"],
            lora_dropout=0.2,
            bias="all",
        )
        lora_model = get_peft_model(base_model, config)
        return accelerator.prepare(lora_model)
    
    data_version = 'v2'      
    task_type    = "multi-label"                          
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = esm_path.split('/')[-1]
    output_dir = f"models/{task_type}/{data_version}/{model_name}_{timestamp_str}"

    os.environ["WANDB_PROJECT"] = f"{task_type}_{data_version}_{timestamp_str}"

    # store the wandb offline
    #os.environ["WANDB_MODE"] = "offline"
    
    args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=25,             
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        save_strategy="epoch",
        weight_decay = 0.0,      
        max_grad_norm = 1.0,    
    )

    trainer = CustomTrainer(
        model=None,
        args=args,
        class_weights=class_weights,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_multi_label_metrics,
        model_init=model_init,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend= "wandb",
        hp_space=wandb_hp_space,
        n_trials=10,
    )
    if best_trial is None:
        print("No best trial found during hyperparameter search.")
        return

    print("Best Trial:", best_trial)

    def train_final_model(best_trial):
        best_hyperparameters = best_trial.hyperparameters
        model = model_init(None)
        args.learning_rate = best_hyperparameters["learning_rate"]  
        args.per_device_train_batch_size = best_hyperparameters["per_device_train_batch_size"]
        args.weight_decay = best_hyperparameters["weight_decay"]
        args.max_grad_norm = best_hyperparameters["max_grad_norm"]
        final_trainer = CustomTrainer(
            model=model,
            args=args,
            class_weights=class_weights,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_multi_label_metrics,
        )
        final_trainer.train()

        bese_model_path = os.path.join(output_dir,'best')
        model.config.save_pretrained(bese_model_path)
        final_trainer.save_model(bese_model_path)

        model.save_pretrained(bese_model_path)
        tokenizer.save_pretrained(bese_model_path)
    
    train_final_model(best_trial)

