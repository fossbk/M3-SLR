import numpy as np
import torch
import math
from einops import rearrange
import torch.nn.functional as F
import torch
from multiprocessing.sharedctypes import Value
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve or validation accuracy doesn't increase after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, 
                 path_loss='checkpoint_loss.pt', path_acc='checkpoint_acc.pt', trace_func=print):
        """
        Args:
            patience (int): Số epoch tối đa không cải thiện sau khi dừng huấn luyện.
                            Mặc định: 7
            verbose (bool): Nếu True, in ra thông báo mỗi khi có sự cải thiện.
                            Mặc định: False
            delta (float): Sự thay đổi tối thiểu trong giá trị được giám sát để xem xét là cải thiện.
                            Mặc định: 0
            path_loss (str): Đường dẫn để lưu checkpoint mô hình dựa trên loss.
                            Mặc định: 'checkpoint_loss.pt'
            path_acc (str): Đường dẫn để lưu checkpoint mô hình dựa trên accuracy.
                            Mặc định: 'checkpoint_acc.pt'
            trace_func (function): Hàm để in thông báo.
                            Mặc định: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter_loss = 0
        self.counter_acc = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path_loss = path_loss
        self.path_acc = path_acc
        self.trace_func = trace_func

    def __call__(self, val_loss, val_acc, model):
        # Kiểm tra cải thiện của validation loss
        loss_improved = False
        acc_improved = False

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint_loss(val_loss, model)
            loss_improved = True
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint_loss(val_loss, model)
            self.counter_loss = 0
            loss_improved = True
        else:
            self.counter_loss += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter (loss): {self.counter_loss} out of {self.patience}')

        # Kiểm tra cải thiện của validation accuracy
        if self.best_acc is None:
            self.best_acc = val_acc
            self.save_checkpoint_acc(val_acc, model)
            acc_improved = True
        elif val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.save_checkpoint_acc(val_acc, model)
            self.counter_acc = 0
            acc_improved = True
        else:
            self.counter_acc += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter (acc): {self.counter_acc} out of {self.patience}')

        # Kiểm tra xem có cần dừng huấn luyện không
        if self.counter_loss >= self.patience and self.counter_acc >= self.patience:
            self.early_stop = True

    def save_checkpoint_loss(self, val_loss, model):
        '''Lưu mô hình khi validation loss giảm.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model (loss) ...')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), self.path_loss)
        else:
            torch.save(model.state_dict(), self.path_loss)
        self.val_loss_min = val_loss

    def save_checkpoint_acc(self, val_acc, model):
        '''Lưu mô hình khi validation accuracy tăng.'''
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model (acc) ...')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), self.path_acc)
        else:
            torch.save(model.state_dict(), self.path_acc)
        self.val_acc_max = val_acc
class MyCustomLoss(nn.Module):
    def __init__(self, label_smoothing=0):
        super(MyCustomLoss, self).__init__()
        print("Use Label Smoothing: ", label_smoothing)
        self.crossentropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

    def classification_loss_mixup(self, logits, labels_a, labels_b, lam, epoch):
        """
        Tính classification loss với mixup:
          loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)
        """
        loss_a = self.crossentropy(logits, labels_a)
        loss_b = self.crossentropy(logits, labels_b)
        loss = lam * loss_a + (1 - lam) * loss_b
        return loss, {'classification_loss': loss.item()}

    def forward(self, logits=None, labels=None, trans_feat_s=None, trans_feat_t=None, 
                student_features=None, teacher_features=None, student_logits=None, teacher_logits=None, 
                visual_fusion_ft=None, gloss_visual_fusion_ft=None, **kwargs):
        loss = 0
        loss_dict = {}

        if trans_feat_t is not None and trans_feat_s is not None:
            mse_loss = self.mse(trans_feat_s, trans_feat_t)
            loss += mse_loss
            loss_dict['mse_loss'] = mse_loss.item()

        if logits is not None:
            classification_loss = self.crossentropy(logits, labels)
            loss += classification_loss
            loss_dict['classification_loss'] = classification_loss.item()

        return loss, loss_dict

class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss