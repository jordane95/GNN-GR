depth = 20

test_labels = {} # Dict[int, List[int]], qid -> pos_ids

with open("data/wq/sbert_test.tsv") as f:
    for line in f:
        qid, pos_ids = line.strip().split('\t')
        qid = int(qid)
        pos_ids = list(map(int, pos_ids.split(',')))
        test_labels[qid] = pos_ids

pred_labels = {}
with open("data/wq/reina_test.tsv", 'r') as f:
    for line in f:
        qid, pos_ids = line.strip().split('\t')
        qid = int(qid)
        pos_ids = list(map(int, pos_ids.split(',')))
        pred_labels[qid] = pos_ids
        
precisions = []
recalls = []
f1s = []
hits = 0
for qid, pred_pids in pred_labels.item():
    pred_pids: List[int]
    qid: int
    true_pids: List[int] = test_labels[qid]
    p = len(set(pred_pids) & set(true_pids)) / len(pred_pids)
    r = len(set(pred_pids) & set(true_pids)) / len(true_pids)
    hits += 0 if r == 0 else 1
    precisions.append(p)
    recalls.append(r)
    f1s.append(2*p*r/(p+r+1e-7))
precision = sum(precisions) / len(precisions)
recall = sum(recalls) / len(recalls)
f1 = sum(f1s) / len(f1s)
accuracy = hits / len(q_lookup)

print(f"Precision@{depth}: {precision}")
print(f"Recall@{depth}: {recall}")
print(f"F1@{depth}: {f1}")
print(f"Accuracy@{depth}: {accuracy}")