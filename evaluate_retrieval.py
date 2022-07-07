import argparse
import yaml
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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


def main():

    test_data = load_results("data/wq/sbert_test.tsv")
    retrieval_results = load_results("data/wq/test_rank.tsv")
    
    depth = 1

    precisions = []
    recalls = []
    f1s = []
    hits = 0
    for qid, pred_pids in retrieval_results.items():
        true_pids = test_data[qid]
        pred_pids = pred_pids[:depth]
        p = len(set(pred_pids) & set(true_pids)) / len(pred_pids)
        r = len(set(pred_pids) & set(true_pids)) / len(true_pids)
        hits += 0 if r == 0 else 1
        precisions.append(p)
        recalls.append(r)
        f1s.append(2*p*r/(p+r+1e-7))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = sum(f1s) / len(f1s)
    accuracy = hits / len(retrieval_results)

    logger.info(f"Precision@{depth}: {precision}")
    logger.info(f"Recall@{depth}: {recall}")
    logger.info(f"F1@{depth}: {f1}")
    logger.info(f"Accuracy@{depth}: {accuracy}")


if __name__ == '__main__':
    main()