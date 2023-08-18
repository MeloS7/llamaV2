import json
import argparse
from datasets import load_dataset

def dataset_intersection(dataset1, dataset2):
    '''
    Keep the intersection of the two datasets by labels
    return: a list of dictionaries
    '''
    intersection = []
    for i, v in enumerate(zip(dataset1, dataset2)):
        # Check if the ids are the same
        assert v[0]['id'] == v[1]['id'], "The ids are not the same"

        label1 = v[0]['User label'].lower()
        label2 = v[1]['User label'].lower()

        if label1 == label2:
            intersection.append(v[0])
        
    return intersection
        

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Do a data intersection by labels between two datasets")

    # Add the arguments to the parser
    parser.add_argument(
        "--dataset1",
        type=str,
        help="The first dataset to be compared",
        required=True
    )

    parser.add_argument(
        "--dataset2",
        type=str,
        help="The second dataset to be compared",
        required=True
    )

    args = parser.parse_args()

    # Load datasets
    root_path = '../data/'
    filename_prefix = 'annotated_data_'
    filename_suffix = '.json'
    dataset1_path = root_path + filename_prefix + args.dataset1 + filename_suffix
    dataset2_path = root_path + filename_prefix + args.dataset2 + filename_suffix
    dataset1 = load_dataset('json', data_files=dataset1_path, split='train')
    dataset2 = load_dataset('json', data_files=dataset2_path, split='train')
    print("=====================================")
    print("Dataset 1: ", args.dataset1)
    print("Dataset 2: ", args.dataset2)

    # Compare datasets and take a intersection
    intersection = dataset_intersection(dataset1, dataset2)
    print("=====================================")
    print("Complete intersection!")
    print("Intersection size: ", len(intersection))

    # Save the intersection as a json file
    filename = root_path + 'daccord_' + args.dataset1 + '_' + args.dataset2 + filename_suffix
    with open(filename, 'w', encoding='utf-8') as f:
        for item in intersection:
            json.dump(item, f)
            f.write("\n")
    print("=====================================")
    print("Saving intersection to: ", filename)
    



if __name__ == "__main__":
    main()