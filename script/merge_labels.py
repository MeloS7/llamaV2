import argparse
import copy
import json
from datasets import load_dataset
from collections import Counter

label_map = {
        1: "Support Gun Polarized",
        2: "Support Gun Non-Polarized",
        3: "Neutral",
        4: "Anti Gun Polarized",
        5: "Anti Gun Non-Polarized",
        6: "Not Relevant",
        7: "Not Sure"
    }

def load_datasets(file_path_1='/annotated_data_manual.json'):
    data_file_folder = '../data'
    dataset_self_annot_path = data_file_folder + file_path_1

    dataset_self_annot = load_dataset('json', data_files=dataset_self_annot_path, split='train')

    return dataset_self_annot

def pre_processing(merge_labels, new_labels):
    merge_labels = eval(merge_labels)
    new_labels = eval(new_labels)

    # Create a new label map
    new_label_map = {}
    for k, v in merge_labels.items():
        for label in v:
            assert label not in new_label_map, "The label {} is already in the new label map!"
            new_label_map[label_map[label].lower()] = new_labels[k].lower()
    
    print("The new label map is: ")
    print(new_label_map)
    for k, v in new_label_map.items():
        print(k, ":", v)
    print("=====================================")
    
    return new_label_map

def merge_func(dataset, new_label_map):
    new_dataset = copy.deepcopy(dataset)

    def map_label(item):
        label = item['User label'].lower()
        assert label in new_label_map, "The label {} is not in the new label map!".format(label)
        item['User label'] = new_label_map[label]
        return item

    new_dataset = new_dataset.map(map_label)

    return new_dataset

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compare two datasets and show the difference one by one")

    path_prefix = '/'
    path_suffix = '.json'
    # Add arguments for file paths
    parser.add_argument(
        "--file_name_1",
        help="Path to the self-annotated dataset file",
        default='daccord_yifei_v2_leo',
        action="store"
    )
    parser.add_argument(
        "--merged_labels", 
        help="The way to merge labels. e.g. 1:[1,2,3] means merge label 2 and 3 into label 1.", 
        default="{1:[1,2], 4:[4,5], 7:[3,6,7]}",
        action="store"
    )
    parser.add_argument(
        "--new_labels",
        help="The new labels after merging. e.g. 1:'Support Gun', 4:'Anti Gun'",
        default="{1:'Support', 4:'Anti', 7:'Others'}",
        action="store"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    file_path_1 = path_prefix + args.file_name_1 + path_suffix
    merge_labels = args.merged_labels
    new_labels = args.new_labels

    # Pre-processing inputs
    new_label_map = pre_processing(merge_labels, new_labels)

    # Load datasets
    dataset_to_merge= load_datasets(file_path_1)
    print("The dataset 1 is loaded from", args.file_name_1)
    print("The length of the dataset is: ", len(dataset_to_merge))
    print("=====================================")

    # Merge labels
    new_dataset = merge_func(dataset_to_merge, new_label_map)

    # Save the new dataset
    filename = "../data/" + args.file_name_1 + "_labelMerged.json"
    with open(filename, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            json.dump(item, f)
            f.write("\n")
    print("=====================================")
    print("Saving intersection to: ", filename)

if __name__ == "__main__":
    main()