import argparse
import yaml
import logging

import torch
import numpy as np
from collections import OrderedDict

from core.model_handler import ModelHandler

from faiss_retriever import BaseFaissIPRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


################################################################################
# Main #
################################################################################


def search_queries(retriever, q_reps, p_lookup, batch_size=-1, depth=100):
    if batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    return all_scores, psg_indices


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    q_reps, q_lookup = model.encode(model.test_loader)
    p_reps, p_lookup = model.encode(model.train_loader)

    retriever = BaseFaissIPRetriever()
    retriever = BaseFaissIPRetriever(p_reps)
    retriever.add(p_reps)

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, p_lookup, batch_size=-1, depth=5)
    logger.info('Index Search Finished')

    test_data = model.test_loader.data # List[Tuple[int, List[int], List[int]]]

    test_labels = {} # Dict[int, List[int]], qid -> pos_ids
    
    for qid, pos_ids, neg_ids in test_data:
        test_labels[qid] = pos_ids
    
    precisions = []
    recalls = []
    f1s = []
    for qid, pred_pids in zip(q_lookup, psg_indices):
        pred_pids: List[int]
        qid: int
        true_pids: List[int] = test_labels[qid]
        p = len(set(pred_pids) & set(true_pids)) / len(pred_pids)
        r = len(set(pred_pids) & set(true_pids)) / len(true_pids)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2*p*r/(p+r+1e-7))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = sum(f1s) / len(f1s)

    logger.info(f"Precision@{args.depth}: {precision}")
    logger.info(f"Recall@{args.depth}: {recall}")
    logger.info(f"F1@{args.depth}: {f1}")
    


def grid_search_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    grid_search_hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)


    best_config = None
    best_metric = None
    best_score = -1
    configs = grid(config)
    for cnf in configs:
        print('\n')
        pretrained = True if cnf['out_dir'] is None else False
        for k in grid_search_hyperparams:
            if pretrained:
                cnf['pretrained'] += '_{}_{}'.format(k, cnf[k])
            else:
                cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])
        if pretrained:
            print(cnf['pretrained'])
        else:
            print(cnf['out_dir'])

        model = ModelHandler(cnf)
        dev_metrics = model.train()
        if best_score < dev_metrics[cnf['eary_stop_metric']]:
            best_score = dev_metrics[cnf['eary_stop_metric']]
            best_config = cnf
            best_metric = dev_metrics
            print('Found a better configuration: {}'.format(best_score))

    print('\nBest configuration:')
    for k in grid_search_hyperparams:
        print('{}: {}'.format(k, best_config[k]))

    print('Best score: {}'.format(best_score))

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        # config = yaml.load(setting, Loader=yaml.FullLoader)
        config = yaml.load(setting)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['grid_search']:
        grid_search_main(config)
    else:
        main(config)
