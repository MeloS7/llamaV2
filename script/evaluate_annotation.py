import argparse
from datasets import load_dataset
from collections import Counter

def load_datasets(file_path_1='/annotated_data_manual.json', file_path_2='/annotated_data_llama2.json'):
    data_file_folder = '../data'
    dataset_self_annot_path = data_file_folder + file_path_1
    dataset_llama_annot_path = data_file_folder + file_path_2

    dataset_self_annot = load_dataset('json', data_files=dataset_self_annot_path, split='train')
    dataset_llama_annot = load_dataset('json', data_files=dataset_llama_annot_path, split='train')

    return dataset_self_annot, dataset_llama_annot

def compare_datasets(dataset_self, dataset_llama):
    res_list = []
    label_diff = {}
    for i,v in enumerate(zip(dataset_self, dataset_llama)):
        self_label = v[0]['User label'].lower()
        llama_label = v[1]['User label'].lower()
        
        if self_label == llama_label:
            res_list.append(1)
        else:
            res_list.append(0)
            label_diff[i] = (self_label, llama_label)

    return res_list, label_diff
    
def showAccuracy(res_list):
    print("Accuracy:", res_list.count(1)/len(res_list))

def showLabelDiff(label_diff):
    print("=====================================")
    print("The different labels between datasets:")
    print(Counter(label_diff.values()))

    print("=====================================")
    print("The most mislabeled labels:")
    labels = []
    for i, v in enumerate(label_diff.items()):
        labels.append(v[1][0])
    print(Counter(labels))
    

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compare two datasets")

    # Add arguments for file paths
    parser.add_argument(
        "--file_path_1",
        help="Path to the self-annotated dataset file",
        default='/annotated_data_manual.json',
        action="store"
    )
    parser.add_argument(
        "--file_path_2", 
        help="Path to the llama-annotated dataset file", 
        default='/annotated_data_llama2.json',
        action="store"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    file_path_1 = args.file_path_1
    file_path_2 = args.file_path_2

    # Load datasets
    dataset_self_annot, dataset_llama_annot = load_datasets(file_path_1, file_path_2)

    # Compare datasets
    res_list, label_diff = compare_datasets(dataset_self_annot, dataset_llama_annot)

    # Print results
    showAccuracy(res_list)
    showLabelDiff(label_diff)

if __name__ == "__main__":
    main()