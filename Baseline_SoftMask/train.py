# -*- coding: utf-8 -*-
import os, sys
# Allow imports from project root (shared/, data_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math, argparse, random, numpy, torch, torch.nn as nn
from shared.bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models.asgcn_softmask import ASGCN_SoftMask
from models import LSTM, ASCNN, ASTCN


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader  = BucketIterator(data=absa_dataset.test_data,  batch_size=opt.batch_size, shuffle=False)
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self.global_f1 = 0.
        n_trainable, n_frozen = 0, 0
        for p in self.model.parameters():
            n = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad: n_trainable += n
            else: n_frozen += n
        print(f'n_trainable_params: {n_trainable}, n_nontrainable_params: {n_frozen}')
        for arg in vars(self.opt): print(f'>>> {arg}: {getattr(self.opt, arg)}')
        if torch.cuda.is_available(): print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1: self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc, max_test_f1 = 0, 0
        global_step, continue_not_increase = 0, 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch:', epoch)
            n_correct, n_total, increase_flag = 0, 0, False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()
                inputs  = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total   += len(outputs)
                    train_acc  = n_correct / n_total
                    test_acc, test_f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc: max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1   = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            os.makedirs('state_dict', exist_ok=True)
                            torch.save(self.model.state_dict(), f'state_dict/{self.opt.model_name}_{self.opt.dataset}.pkl')
                            print('>>> best model saved.')
                    print(f'loss: {loss.item():.4f}, acc: {train_acc:.4f}, test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}')
            if not increase_flag:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

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
                t_targets_all = t_targets if t_targets_all is None else torch.cat((t_targets_all, t_targets), 0)
                t_outputs_all = t_outputs if t_outputs_all is None else torch.cat((t_outputs_all, t_outputs), 0)
        test_acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0,1,2], average='macro')
        return test_acc, f1

    def run(self, repeats=3):
        criterion = nn.CrossEntropyLoss()
        os.makedirs('log', exist_ok=True)
        f_out = open(f'log/{self.opt.model_name}_{self.opt.dataset}_val.txt', 'w', encoding='utf-8')
        max_test_acc_avg = max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat:', i + 1)
            f_out.write(f'repeat: {i+1}\n')
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            print(f'max_test_acc: {max_test_acc:.4f}  max_test_f1: {max_test_f1:.4f}')
            f_out.write(f'max_test_acc: {max_test_acc:.4f}, max_test_f1: {max_test_f1:.4f}\n')
            max_test_acc_avg += max_test_acc
            max_test_f1_avg  += max_test_f1
            print('#' * 100)
        print(f'max_test_acc_avg: {max_test_acc_avg / repeats:.4f}')
        print(f'max_test_f1_avg:  {max_test_f1_avg  / repeats:.4f}')
        f_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',     default='asgcn_softmask',           type=str)
    parser.add_argument('--dataset',        default='rest14',          type=str)
    parser.add_argument('--optimizer',      default='adam',            type=str)
    parser.add_argument('--initializer',    default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate',  default=0.001,  type=float)
    parser.add_argument('--l2reg',          default=0.00001,type=float)
    parser.add_argument('--num_epoch',      default=100,    type=int)
    parser.add_argument('--batch_size',     default=32,     type=int)
    parser.add_argument('--log_step',       default=5,      type=int)
    parser.add_argument('--embed_dim',      default=300,    type=int)
    parser.add_argument('--hidden_dim',     default=300,    type=int)
    parser.add_argument('--polarities_dim', default=3,      type=int)
    parser.add_argument('--repeats',        default=3,      type=int)
    parser.add_argument('--save',           default=False,  type=bool)
    parser.add_argument('--seed',           default=776,    type=int)
    parser.add_argument('--device',         default=None,   type=str)
    opt = parser.parse_args()

    model_classes = {'asgcn_softmask': ASGCN_SoftMask, 'lstm': LSTM, 'ascnn': ASCNN, 'astcn': ASTCN}
    input_colses  = {
        'lstm':   ['text_indices'],
        'ascnn':  ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn_softmask':  ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn':  ['text_indices', 'aspect_indices', 'left_indices', 'dependency_tree'],
    }
    initializers = {'xavier_uniform_': torch.nn.init.xavier_uniform_, 'xavier_normal_': torch.nn.init.xavier_normal_, 'orthogonal_': torch.nn.init.orthogonal_}
    optimizers   = {'adam': torch.optim.Adam, 'adadelta': torch.optim.Adadelta, 'adagrad': torch.optim.Adagrad, 'adamax': torch.optim.Adamax, 'sgd': torch.optim.SGD}

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer   = optimizers[opt.optimizer]
    opt.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed); numpy.random.seed(opt.seed); torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(repeats=opt.repeats)
