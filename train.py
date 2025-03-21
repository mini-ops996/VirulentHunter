import os
from datetime import datetime
import wandb
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,

)
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType

from protein_dataset import ProteinDataset
from utils import parse_inputs, compute_metrics
import wandb


def wandb_hp_space(trial):
    return{
        "method": "random",
        "metric": {"name": "mcc", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-3},# max: 1e-2 will be bad results
            "per_device_train_batch_size": {"values": [4]}, # train and valid all 8 will out of mem
            "weight_decay" :{"values": [0.0,0.2,0.4,0.5,0.6,0.8,1.0]},            # new add
            "max_grad_norm": {"values": [0.0,0.1,0.2,0.5,1.0]},                       # new add
         },
    }


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")   
        outputs = model(**inputs)
        logits = outputs.logits         
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels.long())
        return (loss, outputs) if return_outputs else loss

def train_protein_model(esm_path, input_fasta_path, input_label_path,max_length):

    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    train_sequences, train_labels, val_sequences, val_labels = parse_inputs(input_fasta_path, input_label_path)
    
    train_dataset = ProteinDataset(train_sequences, train_labels, tokenizer, max_length)
    val_dataset =   ProteinDataset(val_sequences, val_labels, tokenizer, max_length)


    accelerator = Accelerator()
    train_dataset, val_dataset = accelerator.prepare(train_dataset, val_dataset)

    def model_init(trial):
        base_model = EsmForSequenceClassification.from_pretrained(
            esm_path, num_labels=2, hidden_dropout_prob=0.2,
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,      
            target_modules=["query", "key", "value"],
            lora_dropout=0.3,
            bias="all",
        )
        lora_model = get_peft_model(base_model, config)
        return accelerator.prepare(lora_model)
    
    data_version = 'v7'      
    task_type    = "binary"                           
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = esm_path.split('/')[-1]
    output_dir = f"models/{task_type}/{data_version}/{model_name}_{timestamp_str}"
    
    os.environ["WANDB_PROJECT"] = f"{task_type}_{data_version}_{timestamp_str}"

    
    args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=30,             
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        save_strategy="epoch",
        weight_decay = 0.0,      
        max_grad_norm = 1.0,    
    )

    trainer = CustomTrainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="wandb",
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
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        final_trainer.train()

        best_model_path = os.path.join(output_dir,'best')
        model.config.save_pretrained(best_model_path)
        final_trainer.save_model(best_model_path)

        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    
    train_final_model(best_trial)

