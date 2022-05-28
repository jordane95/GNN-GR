import os
import random
import argparse
from typing import Dict, List


def load_results(path: str) -> Dict[str, List[str]]:
    """Load intermediate results from disk

    Args:
        path (str): where to read
    Returns:
        results (Dict[str, List[str]]): intermediate results, qid -> docids
    """
    results: Dict[str, List[str]] = {}
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            if len(data) != 2:
                continue
            qid, docids = data
            docids = docids.split(',')
            results[qid] = docids
    return results

def merge(pos_docs: Dict[str, List[str]], neg_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Args:
        pos_docs (Dict[str, List[str]]): results from sbert as positives
        neg_docs (Dict[str, List[str]]): results from reina / bm25 as negs excluding pos
    Returns:
        results (Dict[str, Dict[str, List[str]]]): results containing of triples, qid -> (pos_ids, neg_ids)
    """
    results: Dict[str, Dict[str, List[str]]] = {}
    for qid in pos_docs:
        if qid in neg_docs:
            pos_ids = pos_docs[qid]
            neg_ids = neg_docs[qid]
            neg_ids = list(set(neg_ids) - set(pos_ids))
            results[qid] = {
                "pos_ids": pos_ids,
                "neg_ids": neg_ids,
            }
    return results


def save_results(results: Dict[str, Dict[str, List[str]]], path: str) -> None:
    """Save triples to disk

    Args:
        results (Dict[str, Dict[str, List[str]]]): triples
        path (str): save path
    Returns:
        None
    """
    with open(path, 'w') as f:
        for qid in results:
            pos_ids = results[qid]['pos_ids']
            neg_ids = results[qid]['neg_ids']
            f.write(f"{qid}\t{','.join(pos_ids)}\t{','.join(neg_ids)}\n")
    

def read_triples(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Read triples from path

    Args:
        path (str): path that stores triples
    Returns:
        results (Dict[str, Dict[str, List[str]]]): triples
    """
    results: Dict[str, Dict[str, List[str]]] = {}
    with open(path, 'r') as f:
        for line in f:
            qid, pos_ids, neg_ids = line.strip().split('\t')
            pos_ids = pos_ids.split(',')
            neg_ids = neg_ids.split(',')
            results[qid] = {
                "pos_ids": pos_ids,
                "neg_ids": neg_ids,
            }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge retrieval results from sbert and reina to construct positive and negative pairs for retrieval.')
    parser.add_argument('--data_path', type=str, default='data/wq', help='Path to the dataset')

    args = parser.parse_args()

    for set_name in ['train', 'dev', 'test']:
        pos_path = os.path.join(args.data_path, f"sbert_{set_name}.tsv")
        neg_path = os.path.join(args.data_path, f"reina_{set_name}.tsv")
        pos_results = load_results(pos_path)
        neg_results = load_results(neg_path)
        triples = merge(pos_results, neg_results)
        save_path = os.path.join(args.data_path, f"{set_name}.tsv")
        save_results(triples, save_path)
