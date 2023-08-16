import torch
import re
import os
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        val = self.dataset[idx]
        label_text = val['label_text']
        sentence = val['text']  # Here, sentence is already in the format of llama prompt tamplate
   
        inputs = self.tokenizer(sentence, return_tensors="pt").to("cuda")
        labels = self.tokenizer(label_text, return_tensors="pt").to("cuda")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def collate_fn(batch):
    input_ids = [item['input_ids'].tolist() for item in batch]
    attention_mask = [item['attention_mask'].tolist() for item in batch]
    labels = [item['labels'] for item in batch]

    # Left Padding
    max_length = max([len(item) for item in input_ids])
    input_ids = [[0]*(max_length - len(item)) + item for item in input_ids]
    attention_mask = [[0]*(max_length - len(item)) + item for item in attention_mask]

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    # Usually, labels are not padded
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def evaluate_SFT(peft_model, dataset, tokenizer, batch_size=16):
    chatDataset = CustomDataset(dataset, tokenizer)
    data_loader = DataLoader(chatDataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    peft_model.eval()

    compared_result = []
    invalid_label = []

    for i, batch in enumerate(data_loader):
        print("Batch {}/{}".format(i+1, len(data_loader)/batch_size))
        # Move batch to GPU
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        # Generate for the entire batch
        outputs = peft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated text and labels
        outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        label_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Evaluate the generated text
        for idx in range(len(outputs_text)):
            # Extract the last sentence
            selected_sentiment = outputs_text[idx].split("Sentiment: ")[1].strip()
            selected_sentiment = selected_sentiment.split(" ")[1].lower()
            
            if selected_sentiment not in ['positive', 'negative']:
                invalid_label.append(selected_sentiment)
                compared_result.append(0)
                continue
            
            if selected_sentiment == label_decoded[idx]:
                compared_result.append(1)
            else:
                compared_result.append(0)

        # if i >= 5:
        #     break
        # else:
        #     print("Batch", i, "done!")

    return compared_result, invalid_label

def showEvalResults(compare_results, invalid_label):
    counted_elements = Counter(invalid_label)
    accuracy = compare_results.count(1)/len(compare_results)
    print("Accuracy:", accuracy)
    print("# of Invalid labels:", len(invalid_label), "out of", len(compare_results), "samples")
    print("Invalid labels:", counted_elements)


def main():
    # Read arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--adapters_name", type=str, default="Llama-2-13b-hf-steps")
    
    args = parser.parse_args()

    batch_size = args.batch_size
    base_model_name = args.base_model_name
    adapters_name = "./models/meta-llama/" + args.adapters_name

    # Login to Hugging Face
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError("Hugging Face token not provided!")

    login(token=HUGGINGFACE_TOKEN)
    print("Successfully logged in!")
    print("=====================================")

    # Load the validation dataset
    dataset_name = "OneFly7/llama2-sst2-fine-tuning-without-system-info"
    validataion_dataset = load_dataset(dataset_name, split="validation")

    ## Version 2-7b for finetuning
    # base_model_name = "meta-llama/Llama-2-13b-hf"
    # adapters_name = "./models/llama-2-13b-hf-SFT-5000-1ep"

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


    peft_model = PeftModel.from_pretrained(base_model, adapters_name)
    # This method merges the LoRa layers into the base model. 
    # This is needed if someone wants to use the base model as a standalone model.
    # peft_model = peft_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Evaluate the model
    print("Evaluating the model " + adapters_name + " ...")
    comp_res, invalid_label = evaluate_SFT(peft_model, validataion_dataset, tokenizer, batch_size)
    print("Evaluation completed!")

    # Show the results
    showEvalResults(comp_res, invalid_label)

if __name__ == "__main__":
    main()