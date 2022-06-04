import os
import json
import argparse
from typing import List, Dict


def preprocess_data(data_path: str, qrels_path: str, save_path: str, id2text: Dict[int, str]):
    """
    Args:
        data_path (str): path to the raw dataset
        qrels_path (str): graph retrieval results composed of ids
        save_path (str): path to save the processed data
    Returns:
        None
    """
    # load raw data
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    # raw_data (List[Dict]): list of data points

    # load qrels
    qrels: Dict[int, List[int]] = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, docids = line.strip().split('\t')
            qid = int(qid)
            docids = list(map(int, docids.split(','))) # List[int]
            qrels[qid] = docids
    
    # prepare new data and save to destination path
    for data in raw_data:
        qid = data['id']
        docids = qrels[qid] # List[int]
        refs = [id2text[docid] for docid in docids] # List[str]
        data['refs'] = refs
    
    with open(save_path, 'w') as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/wq')

    args = parser.parse_args()

    id2text: Dict[int, str] = {}
    with open(os.path.join(args.data_path, 'train.json'), 'r') as f:
        train_data = json.load(f)
    # construct a id -> text dict
    for data in train_data:
        id2text[data['id']] = data['text'][0]

    for set_name in ['train', 'dev', 'test']:
        raw_data_path = os.path.join(args.data_path, f"{set_name}.json")
        qrels_path = os.path.join(args.data_path, f"{set_name}_rank.tsv")
        save_path = os.path.join(args.data_path, f"{set_name}_gr.json")
        preprocess_data(raw_data_path, qrels_path, save_path, id2text)
