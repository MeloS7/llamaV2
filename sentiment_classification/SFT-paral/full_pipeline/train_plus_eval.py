import torch
import re
import os
import argparse
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import evaluate_SFT, showEvalResults

def main():
    # Read arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--dataset_name", type=str, default="OneFly7/llama2-sst2-fine-tuning-without-system-info")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--stop_condition", type=str, default="steps")
    
    args = parser.parse_args()

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    epoch = args.epoch
    steps = args.steps
    base_model_name = args.base_model_name
    dataset_name = args.dataset_name
    max_seq_length = args.max_seq_length
    stop_condition = args.stop_condition

    # Login to Hugging Face
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError("Hugging Face token not provided!")

    login(token=HUGGINGFACE_TOKEN)
    print("Successfully logged in!")
    print("=====================================")

    # Load the validation dataset
    dataset_name = "OneFly7/llama2-sst2-fine-tuning-without-system-info"
    train_dataset = load_dataset(dataset_name, split="train")
    validataion_dataset = load_dataset(dataset_name, split="validation")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True
    )

    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1 

    # Set peft config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = "./results"

    # Set training arguments
    if stop_condition == "steps":
        print("The training will stop after {} steps.".format(steps))
        print("The number of training examples is {}.".format(train_batch_size*steps))
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=20,
            max_steps=steps,
        )
    elif stop_condition == "epochs":
        print("The training will stop after {} epochs.".format(epoch))
        print("The number of training examples is {}.".format(len(train_dataset)*epoch))
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=20,
            num_train_epochs=epoch,
        )

    ####
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    ####

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )

    # Train the model
    print("Training the model " + base_model_name + "...")
    trainer.train()
    print("Training completed!")
    trainer.save_model("./models/"+base_model_name+"-"+stop_condition)
    print("=====================================")

    # Evaluate the model
    # print("Evaluating the finetuned model " + base_model_name + "...")
    # comp_res, invalid_label = evaluate_SFT(base_model, validataion_dataset, tokenizer, eval_batch_size)
    # print("Evaluation completed!")
    # print("=====================================")

    # Show the results
    showEvalResults(comp_res, invalid_label)

if __name__ == "__main__":
    main()