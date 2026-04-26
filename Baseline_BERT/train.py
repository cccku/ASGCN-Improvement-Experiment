# -*- coding: utf-8 -*-
"""
train.py for Improvement B: BERT-enhanced ASGCN
-------------------------------------------------
Key differences from base train.py:

1. Uses ABSADatesetReaderBert instead of ABSADatesetReader
   → data has extra fields: bert_token_ids, bert_attention_mask, bert_token_type_ids

2. Differential learning rates:
   - BERT encoder weights  → lr = opt.bert_lr      (default 2e-5)
   - Classifier / GCN head → lr = opt.learning_rate (default 1e-3)
   This prevents catastrophic forgetting of BERT's pretrained weights.

3. AdamW optimizer for BERT params (weight decay applied correctly,
   excluding bias and LayerNorm as per the BERT paper).

4. Linear warmup scheduler over the first opt.warmup_steps steps.

Usage:
    python train.py --model_name asgcn_bert --dataset rest14
    python train.py --model_name asgcn_bert --dataset lap14 --bert_lr 3e-5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from bucket_iterator import BucketIterator
from sklearn import metrics

from data_utils_bert import ABSADatesetReaderBert
from models.asgcn_bert import ASGCN_BERT


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReaderBert(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.train_data_loader = BucketIterator(
            data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader  = BucketIterator(
            data=absa_dataset.test_data,  batch_size=opt.batch_size, shuffle=False)

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    # ------------------------------------------------------------------ #
    def _print_args(self):
        n_trainable, n_frozen = 0, 0
        for p in self.model.parameters():
            n = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad: n_trainable += n
            else:               n_frozen    += n
        print(f'n_trainable_params: {n_trainable}, n_nontrainable_params: {n_frozen}')
        print('> training arguments:')
        for arg in vars(self.opt):
            print(f'>>> {arg}: {getattr(self.opt, arg)}')

    # ------------------------------------------------------------------ #
    def _reset_params(self):
        """
        Only reset the non-BERT parameters (GCN layers, fc).
        BERT weights are always restored from pretrained checkpoint via
        from_pretrained, so we leave them untouched here.
        """
        bert_param_names = {n for n, _ in self.model.bert.named_parameters()}
        for name, p in self.model.named_parameters():
            # strip leading "bert." prefix if present
            bare = name[len("bert."):] if name.startswith("bert.") else name
            if bare in bert_param_names:
                continue   # skip BERT params
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # ------------------------------------------------------------------ #
    def _build_optimizer(self):
        """
        Differential learning rates + AdamW with correct weight decay.

        BERT paper recommendation:
          - No weight decay on bias, LayerNorm.weight
          - BERT layers: lr = bert_lr (2e-5)
          - Head (GCN + fc): lr = learning_rate (1e-3)
        """
        no_decay = {'bias', 'LayerNorm.weight'}

        bert_params_decay    = []
        bert_params_no_decay = []
        head_params_decay    = []
        head_params_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_bert = name.startswith('bert.')
            is_no_decay = any(nd in name for nd in no_decay)
            if is_bert:
                if is_no_decay: bert_params_no_decay.append(param)
                else:           bert_params_decay.append(param)
            else:
                if is_no_decay: head_params_no_decay.append(param)
                else:           head_params_decay.append(param)

        param_groups = [
            {'params': bert_params_decay,    'lr': self.opt.bert_lr,      'weight_decay': 0.01},
            {'params': bert_params_no_decay, 'lr': self.opt.bert_lr,      'weight_decay': 0.0},
            {'params': head_params_decay,    'lr': self.opt.learning_rate,'weight_decay': self.opt.l2reg},
            {'params': head_params_no_decay, 'lr': self.opt.learning_rate,'weight_decay': 0.0},
        ]
        optimizer = AdamW(param_groups, eps=1e-8)
        return optimizer

    # ------------------------------------------------------------------ #
    def _build_scheduler(self, optimizer, num_training_steps):
        """Linear warmup then linear decay."""
        warmup_steps = self.opt.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------ #
    def _train(self, criterion, optimizer, scheduler):
        max_test_acc, max_test_f1 = 0, 0
        global_step = 0
        continue_not_increase = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch:', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False

            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()

                inputs  = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping — essential for BERT fine-tuning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total   += len(outputs)
                    train_acc  = n_correct / n_total
                    test_acc, test_f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1   = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            os.makedirs('state_dict', exist_ok=True)
                            torch.save(self.model.state_dict(),
                                       f'state_dict/{self.opt.model_name}_{self.opt.dataset}.pkl')
                            print('>>> best model saved.')
                    print(f'loss: {loss.item():.4f}, acc: {train_acc:.4f}, '
                          f'test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}')

            if not increase_flag:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0

        return max_test_acc, max_test_f1

    # ------------------------------------------------------------------ #
    def _evaluate_acc_f1(self):
        self.model.eval()
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs  = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total   += len(t_outputs)
                t_targets_all = t_targets if t_targets_all is None \
                    else torch.cat((t_targets_all, t_targets), 0)
                t_outputs_all = t_outputs if t_outputs_all is None \
                    else torch.cat((t_outputs_all, t_outputs), 0)
        test_acc = n_correct / n_total
        f1 = metrics.f1_score(
            t_targets_all.cpu(),
            torch.argmax(t_outputs_all, -1).cpu(),
            labels=[0, 1, 2], average='macro')
        return test_acc, f1

    # ------------------------------------------------------------------ #
    def run(self, repeats=1):
        criterion = nn.CrossEntropyLoss()
        os.makedirs('log', exist_ok=True)
        f_out = open(f'log/{self.opt.model_name}_{self.opt.dataset}_val.txt', 'w', encoding='utf-8')

        # Total training steps across all repeats (for scheduler)
        steps_per_epoch    = 1000
        num_training_steps = steps_per_epoch * self.opt.num_epoch * repeats

        max_test_acc_avg = max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat:', i + 1)
            f_out.write(f'repeat: {i+1}\n')
            self._reset_params()

            optimizer = self._build_optimizer()
            scheduler = self._build_scheduler(optimizer, num_training_steps)

            max_test_acc, max_test_f1 = self._train(criterion, optimizer, scheduler)
            print(f'max_test_acc: {max_test_acc:.4f}  max_test_f1: {max_test_f1:.4f}')
            f_out.write(f'max_test_acc: {max_test_acc:.4f}, max_test_f1: {max_test_f1:.4f}\n')
            max_test_acc_avg += max_test_acc
            max_test_f1_avg  += max_test_f1
            print('#' * 100)

        print(f'max_test_acc_avg: {max_test_acc_avg / repeats:.4f}')
        print(f'max_test_f1_avg:  {max_test_f1_avg  / repeats:.4f}')
        f_out.close()


# ======================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',    default='asgcn_bert', type=str)
    parser.add_argument('--dataset',       default='rest14',     type=str)
    parser.add_argument('--initializer',   default='xavier_uniform_', type=str)
    # Two learning rates: BERT encoder vs classifier head
    parser.add_argument('--bert_lr',       default=2e-5,   type=float,
                        help='Learning rate for BERT encoder layers')
    parser.add_argument('--learning_rate', default=1e-3,   type=float,
                        help='Learning rate for GCN + classifier head')
    parser.add_argument('--l2reg',         default=0.00001,type=float)
    parser.add_argument('--warmup_steps',  default=100,    type=int,
                        help='Linear warmup steps for the LR scheduler')
    parser.add_argument('--num_epoch',     default=20,     type=int,
                        help='BERT fine-tuning converges faster than GloVe; 20 epochs is usually enough')
    parser.add_argument('--batch_size',    default=16,     type=int,
                        help='Smaller batch for BERT due to GPU memory')
    parser.add_argument('--log_step',      default=5,      type=int)
    parser.add_argument('--embed_dim',     default=300,    type=int)
    parser.add_argument('--hidden_dim',    default=300,    type=int)
    parser.add_argument('--polarities_dim',default=3,      type=int)
    parser.add_argument('--save',          default=False,  type=bool)
    parser.add_argument('--seed',          default=776,    type=int)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--device',        default=None,   type=str)
    opt = parser.parse_args()

    opt.model_class = ASGCN_BERT
    opt.inputs_cols = [
        'bert_token_ids',
        'bert_attention_mask',
        'bert_token_type_ids',
        'aspect_indices',
        'left_indices',
        'dependency_graph',
    ]
    opt.initializer = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_':  torch.nn.init.xavier_normal_,
        'orthogonal_':     torch.nn.init.orthogonal_,
    }[opt.initializer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
                 if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    ins = Instructor(opt)
    ins.run(repeats=opt.repeats)