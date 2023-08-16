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


