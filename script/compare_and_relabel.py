import argparse
import copy
import json
from datasets import load_dataset
from collections import Counter

def load_datasets(file_path_1='/annotated_data_manual.json', file_path_2='/annotated_data_llama2.json'):
    data_file_folder = '../data'
    dataset_self_annot_path = data_file_folder + file_path_1
    dataset_llama_annot_path = data_file_folder + file_path_2

    dataset_self_annot = load_dataset('json', data_files=dataset_self_annot_path, split='train')
    dataset_llama_annot = load_dataset('json', data_files=dataset_llama_annot_path, split='train')

    return dataset_self_annot, dataset_llama_annot

def compare_datasets(dataset1, dataset2):
    label_map = {
        1: "Support Gun Polarized",
        2: "Support Gun Non-Polarized",
        3: "Neutral",
        4: "Anti Gun Polarized",
        5: "Anti Gun Non-Polarized",
        6: "Not Relevant",
        7: "Not Sure"
    }

    new_dataset = copy.deepcopy(dataset1)

    for i, v in enumerate(zip(dataset1, dataset2)):
        assert v[0]['id'] == v[1]['id']

        label1 = v[0]['User label'].lower()
        label2 = v[1]['User label'].lower()

        if label1 != label2:
            print("Sentence: ", v[0]['body_cleaned'])
            print("Dataset1 label: ", label1)
            print("Dataset2 label: ", label2)
            print("=====================================")
            print("You have the following options:")
            print("""1. Support Gun Polorized
2. Support Gun Non-Polarized
3. Neutral
4. Anti Gun Polarized
5. Anti Gun Non-Polarized
6. Not Relevant
7. Not Sure
""")
            print("=====================================")
            print("Please enter the correct label number:")
            correct_label = label_map[int(input())]
            print("Your choice: ", correct_label)
            new_dataset[i]['User label'] = correct_label
            print("=====================================")

    return new_dataset


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compare two datasets and show the difference one by one")

    path_prefix = '/annotated_data_'
    path_suffix = '.json'
    # Add arguments for file paths
    parser.add_argument(
        "--file_name_1",
        help="Path to the self-annotated dataset file",
        default='yifei',
        action="store"
    )
    parser.add_argument(
        "--file_name_2", 
        help="Path to the llama-annotated dataset file", 
        default='llama2',
        action="store"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    file_path_1 = path_prefix + args.file_name_1 + path_suffix
    file_path_2 = path_prefix + args.file_name_2 + path_suffix

    # Load datasets
    dataset_self_annot, dataset_llama_annot = load_datasets(file_path_1, file_path_2)
    print("The dataset 1 is loaded from", args.file_name_1)
    print("The dataset 2 is loaded from", args.file_name_2)
    print("=====================================")

    # Compare and relabel datasets
    new_dataset = compare_datasets(dataset_self_annot, dataset_llama_annot)

    # Save the new dataset
    the_file_name = "../data/" + args.file_name_1 + "_relabelled.json"
    with open(the_file_name, 'w') as f:
        json.dump(new_dataset, f)


if __name__ == "__main__":
    main()