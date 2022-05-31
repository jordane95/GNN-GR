'''
Created on Oct, 2019

@author: hugo

'''
import os
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from .models.seq2seq import Seq2Seq
from .models.graph2seq import Graph2Seq
from .utils.vocab_utils import VocabModel
from .utils import constants as Constants
from .utils.generic_utils import to_cuda, create_mask
from .utils.io_utils import load_ndjson
from .evaluation.eval import QGEvalCap, WMD
from .utils.constants import INF
from .layers.common import dropout



class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    def __init__(self, config, train_set=None):
        self.config = config
        if config['model_name'] == 'graph2seq':
            self.net_module = Graph2Seq
        elif config['model_name'] == 'seq2seq':
            self.net_module = Seq2Seq
        else:
            raise RuntimeError('Unknown model_name: {}'.format(config['model_name']))
        print('[ Running {} model ]'.format(config['model_name']))

        self.vocab_model = VocabModel.build(self.config['saved_vocab_file'], train_set, config)
        if config['kg_emb']:
            self.config['num_entities'] = len(self.vocab_model.node_vocab)
            self.config['num_entity_types'] = len(self.vocab_model.node_type_vocab)
            self.config['num_relations'] = len(self.vocab_model.edge_type_vocab)
        else:
            self.vocab_model.node_vocab = None
            self.vocab_model.node_type_vocab = None
            self.vocab_model.edge_type_vocab = None


        if self.config['pretrained']:
            state_dict_opt = self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            for name, p in self.config['bert_model'].named_parameters():
                print('{}: {}'.format(name, str(p.size())))
                num_params += p.numel()
        print('#Parameters = {}\n'.format(num_params))

        self.criterion = nn.NLLLoss(ignore_index=self.vocab_model.word_vocab.PAD)
        self._init_optimizer()


        if config['rl_wmd_ratio'] > 0:
            self.wmd = WMD(config.get('wmd_emb_file', None))
        else:
            self.wmd = None


    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'])
        self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)

        # Merge the arguments
        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            self.config['bert_model'].load_state_dict(state_dict['bert'])

        return state_dict.get('optimizer', None) if state_dict else None

    def _init_new_network(self):
        w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                           pretrained_vecs=self.vocab_model.word_vocab.embeddings)

        if self.config['kg_emb']:
            pretrained_entity_vecs, pretrained_relation_vecs = self.load_pretrained_kg_emb(self.vocab_model.node_vocab, self.vocab_model.edge_type_vocab)


            entity_emb = self._init_embedding(len(self.vocab_model.node_vocab), self.config['entity_emb_dim'],
                                            pretrained_vecs=pretrained_entity_vecs)
            relation_emb = self._init_embedding(len(self.vocab_model.edge_type_vocab), self.config['relation_emb_dim'],
                                            pretrained_vecs=pretrained_relation_vecs)
            del pretrained_entity_vecs
            del pretrained_relation_vecs

        else:
            entity_emb = None
            relation_emb = None

        self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab, entity_emb=entity_emb, relation_emb=relation_emb)

    def load_pretrained_kg_emb(self, node_vocab, edge_type_vocab):
        ent2vec = load_ndjson(self.config['pretrained_entity_embed_file'], return_type='dict')

        pretrained_entity_vecs = []
        for entity in node_vocab.index2word:
            if entity in ent2vec:
                pretrained_entity_vecs.append(ent2vec[entity])
            else:
                pretrained_entity_vecs.append(np.array(np.random.uniform(low=-0.1, high=0.1, size=(self.config['entity_emb_dim'],)), dtype=np.float32))
        pretrained_entity_vecs = np.array(pretrained_entity_vecs)


        rel2vec = load_ndjson(self.config['pretrained_relation_embed_file'], return_type='dict')

        pretrained_relation_vecs = []
        for relation in edge_type_vocab.index2word:
            if relation in rel2vec:
                pretrained_relation_vecs.append(rel2vec[relation])
            else:
                pretrained_relation_vecs.append(np.array(np.random.uniform(low=-0.1, high=0.1, size=(self.config['relation_emb_dim'],)), dtype=np.float32))
        pretrained_relation_vecs = np.array(pretrained_relation_vecs)
        return pretrained_entity_vecs, pretrained_relation_vecs

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, \
                    patience=2, verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'bert': self.config['bert_model'].state_dict() if self.config['use_bert'] and self.config.get('finetune_bert', None) else None,
                'optimizer': self.optimizer.state_dict()
            },
            'config': self.config,
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def predict(self, batch, step, forcing_ratio=1, rl_ratio=0, update=True, out_predictions=False, mode='train'):
        """
        Args:
          batch (Dict[str, Dict[str, object]]): vectorized triple dict input
          ...
        """
        self.network.train(update)

        if mode == 'train':
            loss, loss_value, metrics = train_batch(batch, self.network, self.vocab_model.word_vocab, self.criterion, forcing_ratio, rl_ratio, self.config, wmd=self.wmd)

            # Accumulate gradients
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            # Run backward
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                if self.config['grad_clipping']:
                    # Clip gradients
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    if self.config['use_bert'] and self.config.get('finetune_bert', None):
                        parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

        elif mode == 'dev':
            loss_value, metrics = dev_batch(batch, self.network, self.vocab_model.word_vocab, criterion=None, show_cover_loss=self.config['show_cover_loss'])

        else:
            query_reps, query_ids = test_batch(batch, self.network, self.vocab_model.word_vocab, self.config)
            loss_value = None

        output = {
            'loss': loss_value,
            'metrics': metrics
        }

        if mode == 'test' and out_predictions:
            output['query_reps'] = query_reps
            output['query_ids'] = query_ids
        return output


# Training phase
def train_batch(batch, network, vocab, criterion, forcing_ratio, rl_ratio, config, wmd=None):
    """
    Args:
      batch (Dict[str, Dict[str, object]]): triples
    """
    network.train(True)

    batch_size = batch['batch_size']

    graph_output = {}

    for key in ['query', 'pos', 'neg']:
      graph_batch = batch[key]
      with torch.set_grad_enabled(True):
          ext_vocab_size = graph_batch['oov_dict'].ext_vocab_size if graph_batch['oov_dict'] else None

          network_out = network(graph_batch, criterion=criterion,
                  forcing_ratio=forcing_ratio, partial_forcing=config['partial_forcing'], \
                  sample=config['sample'], ext_vocab_size=ext_vocab_size, \
                  include_cover_loss=config['show_cover_loss'])
          graph_output[key] = network_out
    
    pooler = lambda x: (lambda x: x[0] if network.rnn_type == 'lstm' else x)(x).squeeze(0)
    
    query_reps = pooler(graph_output['query'].encoder_state) # (batch_size, graph_emb_dim)
    pos_reps = pooler(graph_output['pos'].encoder_state)
    neg_reps = pooler(graph_output['neg'].encoder_state)
    
    doc_reps = torch.cat((pos_reps, neg_reps)) # (batch_size*2, graph_emb_dim)

    scores = torch.matmul(query_reps, doc_reps.transpose(0, 1)) # (batch_size, batch_size * 2)
    scores = scores.view(batch_size, -1)

    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)

    loss = nn.CrossEntropyLoss(reduction='mean')(scores, target)

    loss_value = loss.item()

    predict = scores.argmax(dim=1)
    accuracy = sum(predict == target) / batch_size

    metrics = {"Bleu_4": accuracy}

    return loss, loss_value, metrics

# Development phase
def dev_batch(batch, network, vocab, criterion=None, show_cover_loss=False):
  """Test the `network` on the `batch`, return the ROUGE score and the loss."""
  network.train(False)
  batch_size = batch['batch_size']

  graph_output = {}

  for key in ['query', 'pos', 'neg']:
    graph_batch = batch[key]
    with torch.no_grad():
        ext_vocab_size = graph_batch['oov_dict'].ext_vocab_size if graph_batch['oov_dict'] else None

        network_out = network(graph_batch, criterion=criterion, ext_vocab_size=ext_vocab_size, include_cover_loss=show_cover_loss)
        graph_output[key] = network_out
  
  pooler = lambda x: (lambda x: x[0] if network.rnn_type == 'lstm' else x)(x).squeeze(0)
  
  query_reps = pooler(graph_output['query'].encoder_state) # (batch_size, graph_emb_dim)
  pos_reps = pooler(graph_output['pos'].encoder_state)
  neg_reps = pooler(graph_output['neg'].encoder_state)
  
  doc_reps = torch.cat((pos_reps, neg_reps)) # (batch_size*2, graph_emb_dim)

  scores = torch.matmul(query_reps, doc_reps.transpose(0, 1)) # (batch_size, batch_size * 2)
  scores = scores.view(batch_size, -1)

  target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)

  loss = nn.CrossEntropyLoss(reduction='mean')(scores, target)
  loss_value = loss.item()

  predict = scores.argmax(dim=1)
  accuracy = sum(predict == target) / batch_size
  metrics = {"Bleu_4": accuracy}

  return loss_value, metrics


# Testing phase
def test_batch(batch, network, vocab, config):
    network.train(False)
    batch_size = batch['batch_size']

    graph_batch = batch['query']
    with torch.no_grad():
        ext_vocab_size = graph_batch['oov_dict'].ext_vocab_size if graph_batch['oov_dict'] else None

        network_out = network(graph_batch, criterion=criterion, ext_vocab_size=ext_vocab_size, include_cover_loss=show_cover_loss)
    
    pooler = lambda x: (lambda x: x[0] if network.rnn_type == 'lstm' else x)(x).squeeze(0)
    
    query_reps = pooler(network_out.encoder_state) # (batch_size, graph_emb_dim)
    query_ids = batch['qids'] # List[int]
    return query_reps, query_ids


class Hypothesis(object):
  def __init__(self, tokens, log_probs, dec_state, dec_hiddens, enc_attn_weights, num_non_words, rnn_type):
    self.tokens = tokens  # type: List[int]
    self.log_probs = log_probs  # type: List[float]
    self.dec_state = dec_state  # shape: (1, 1, hidden_size)
    self.dec_hiddens = dec_hiddens  # list of dec_hidden_state
    self.enc_attn_weights = enc_attn_weights  # list of shape: (1, 1, src_len)
    self.num_non_words = num_non_words  # type: int
    self.rnn_type = rnn_type

  def __repr__(self):
    return repr(self.tokens)

  def __len__(self):
    return len(self.tokens) - self.num_non_words

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.log_probs)

  def create_next(self, token, log_prob, dec_state, add_dec_states, enc_attn, non_word):
    dec_hidden_state = dec_state[0] if self.rnn_type == 'lstm' else dec_state
    return Hypothesis(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob],
                      dec_state=dec_state, dec_hiddens=
                      self.dec_hiddens + [dec_hidden_state] if add_dec_states else self.dec_hiddens,
                      enc_attn_weights=self.enc_attn_weights + [enc_attn]
                      if enc_attn is not None else self.enc_attn_weights,
                      num_non_words=self.num_non_words + 1 if non_word else self.num_non_words,
                      rnn_type=self.rnn_type)

def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    QGEval = QGEvalCap(eval_targets, eval_predictions)
    scores = QGEval.evaluate()
    return scores



