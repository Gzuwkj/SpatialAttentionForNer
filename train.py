import json
import os
import re
import warnings
import argparse

import random
from abc import ABC
os.environ["GIT_PYTHON_REFRESH"] = 'quiet'

from fastNLP.core import apply_to_collection

if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import numpy as np
import torch
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer, Evaluator
from fastNLP import TorchGradClipCallback, MoreEvaluateCallback
from fastNLP import FitlogCallback, RichCallback, Callback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog


class Progress(RichCallback):

    def __init__(self, print_every: int = 1, loss_round_ndigit: int = 6, monitor: str = None,
                 larger_better: bool = True,
                 format_json=True):
        super().__init__(format_json=format_json, monitor=monitor, larger_better=larger_better, print_every=print_every,
                         loss_round_ndigit=loss_round_ndigit)
        self.step = 0

    def on_after_trainer_initialized(self, trainer, driver):
        if not self.progress_bar.disable:
            self.progress_bar.set_disable(flag=trainer.driver.get_local_rank() != 0)
        super(RichCallback, self).on_after_trainer_initialized(trainer, driver)

    def on_train_begin(self, trainer):
        pass
        # self.task2id['epoch'] = self.progress_bar.add_task(description=f'Epoch:{trainer.cur_epoch_idx}',
        #                                                    total=trainer.n_epochs,
        #                                                    completed=trainer.global_forward_batches/(trainer.n_batches+1e-6)*
        #                                                    trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every / (trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(self.task2id['batch'], completed=trainer.batch_idx_in_epoch)
        else:
            self.task2id['batch'] = self.progress_bar.add_task(description=f'Batch:{trainer.batch_idx_in_epoch}',
                                                               total=trainer.num_batches_per_epoch,
                                                               completed=trainer.batch_idx_in_epoch)

    def on_train_epoch_end(self, trainer):
        self.clear_tasks()
        # self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
        #                          advance=None, completed=trainer.cur_epoch_idx, refresh=True)

    def on_train_end(self, trainer):
        super(RichCallback, self).on_train_end(trainer)
        self.clear_tasks()

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            self.step += 1
            loss = self.loss / self.step
            self.progress_bar.update(self.task2id['batch'], description=f'Batch:{trainer.batch_idx_in_epoch}',
                                     advance=self.print_every,
                                     post_desc=f'Loss:{round(loss, self.loss_round_ndigit)}', refresh=True)
            # self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
            #                          advance=self.epoch_bar_update_advance, refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        rule_style = ''
        text_style = ''
        characters = '-'
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    rule_style = 'spring_green3'
                    text_style = '[bold]'
                    characters = '+'
        self.progress_bar.print()
        self.progress_bar.console.rule(text_style + f"Eval. results on Epoch:{trainer.cur_epoch_idx}, "
                                                    f"Batch:{trainer.batch_idx_in_epoch}",
                                       style=rule_style, characters=characters)
        results = {key: trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
            self.progress_bar.console.print_json(results)
        else:
            self.progress_bar.print(results)

    def clear_tasks(self):
        for key, taskid in self.task2id.items():
            self.progress_bar.remove_task(taskid)
        self.progress_bar.stop()
        self.task2id = {}
        self.loss = 0
        self.step = 0

    @property
    def name(self):  # progress bar的名称
        return 'rich'


class CanItemDataType(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is CanItemDataType:
            item = getattr(subclass, 'item', None)
            return callable(item)
        return NotImplemented

# fitlog.debug()

from model.model import CNNNer
from model.metrics import NERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace05', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--fp16', default=False, type=bool)
parser.add_argument('--thres', default=0.5, type=float)

args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if 'genia' in args.dataset_name:
        args.model_name = 'dmis-lab/biobert-v1.1'
    else:
        # args.model_name = 'bert-large-uncased'
        args.model_name = 'roberta-large'

n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-2
size_embed_dim = 25


######hyper
def set_random_seed(seed: int) -> object:
    """set seeds for reproducibility
    :rtype: object
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fitlog.set_log_dir('logs/')
seed = fitlog.set_rng_seed(rng_seed=args.seed)
# os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
set_random_seed(args.seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, args.model_name)


def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    dls[name] = _dl

model = CNNNer(args.model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
               size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,
               kernel_size=args.kernel_size, n_head=n_head, cnn_depth=args.cnn_depth)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections

counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')

for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])
# callbacks
callbacks = [Progress()]
callbacks.append(FitlogCallback())
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))

evaluate_dls = {}
if 'dev' in dls:
    evaluate_dls = {'dev': dls.get('dev')}
if 'test' in dls:
    evaluate_dls['test'] = dls['test']

allow_nested = True
metrics = {
    'f': NERMetric(matrix_segs=matrix_segs, ent_thres=args.thres, allow_nested=allow_nested)
}

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=0,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor='f#f#test',
                  evaluate_every=-1,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=args.fp16,
                  progress_bar=None
                  )
trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)
fitlog.finish()
