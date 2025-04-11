import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm
from .tools import EarlyStopping
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import pickle
from itertools import chain

class Trainer:
    def __init__(self, model, criterion, optimizer, device, scheduler=None, top_k=5,
                 epoch=100, logging=None, cfg=None, num_accumulation_steps=1,
                 patience=7, verbose=True, delta=0, is_early_stopping=True, gradient_clip_val=1.0,
                 log_train_step=True, log_steps=100, evaluate_strategy="epoch", evaluate_step=50,
                 wandb=None, k_fold=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.top_k = top_k
        self.epoch = epoch
        self.train_acc, self.val_acc = 0, 0
        self.train_losses, self.val_losses, self.train_accs, self.val_accs = [], [], [], []
        self.lr_progress = []
        self.top_train_acc, self.top_val_acc = 0, 0
        self.logging = logging
        self.cfg = cfg
        self.num_accumulation_steps = num_accumulation_steps

        if k_fold is None:
            self.early_stopping = EarlyStopping(
                patience=patience,
                verbose=verbose,
                delta=delta,
                path_loss=f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + "/best_checkpoints_loss.pth",
                path_acc=f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + "/best_checkpoints_acc.pth"
            )
        else:
            self.early_stopping = EarlyStopping(
                patience=patience,
                verbose=verbose,
                delta=delta,
                path_loss=f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{k_fold}_loss.pth",
                path_acc=f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{k_fold}_acc.pth"
            )
        
        self.is_early_stopping = is_early_stopping
        self.gradient_clip_val = gradient_clip_val
        self.log_train_step = log_train_step
        self.log_steps = log_steps
        self.evaluate_strategy = evaluate_strategy
        self.evaluate_step = evaluate_step
        self.wandb = wandb
        self.k_fold = k_fold
        self.test_accuracy = None

    def save_checkpoint(self, model, path):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        if isinstance(x, dict):
            if 'clip' in x:
                tensor = x['clip']
                batch_size = tensor.size(0)
                if use_cuda:
                    index = torch.randperm(batch_size).to(tensor.device)
                else:
                    index = torch.randperm(batch_size)
                mixed_tensor = lam * tensor + (1 - lam) * tensor[index, :]
                x['clip'] = mixed_tensor
            elif all(key in x for key in ['rgb_left', 'rgb_center', 'rgb_right']):
                batch_size = x['rgb_left'].size(0)
                if use_cuda:
                    index = torch.randperm(batch_size).to(x['rgb_left'].device)
                else:
                    index = torch.randperm(batch_size)
                x['rgb_left'] = lam * x['rgb_left'] + (1 - lam) * x['rgb_left'][index, :]
                x['rgb_center'] = lam * x['rgb_center'] + (1 - lam) * x['rgb_center'][index, :]
                x['rgb_right'] = lam * x['rgb_right'] + (1 - lam) * x['rgb_right'][index, :]
            else:
                raise ValueError("Could not identify the appropriate keys in the input dict to apply mixup.")
        else:
            batch_size = x.size(0)
            if use_cuda:
                index = torch.randperm(batch_size).to(x.device)
            else:
                index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def recursive_to_device(self, data, device):
        if torch.is_tensor(data):
            return data.to(device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self.recursive_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.recursive_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.recursive_to_device(item, device) for item in data)
        else:
            return data

    def train(self, train_loader, val_loader, test_loader):
        cfg = self.cfg
        for epoch in tqdm(range(self.epoch)):
            if val_loader is None and test_loader is None:
                train_loss_log, train_loss, _, _, train_acc = self.train_epoch_maskfeat(train_loader, epoch=epoch)
                self.train_losses.append(train_loss / len(train_loader))
            elif cfg['training'].get('use_gsam', False):
                train_loss_log, train_loss, _, _, train_acc = self.train_epoch_gsam(train_loader, epoch=epoch)
                self.train_losses.append(train_loss / len(train_loader))
                self.train_accs.append(train_acc)
            else:
                if self.evaluate_strategy == 'epoch':
                    train_loss_log, train_loss, _, _, train_acc = self.train_epoch(train_loader, epoch=epoch)
                    self.train_losses.append(train_loss / len(train_loader))
                    self.train_accs.append(train_acc)
                else:
                    train_loss_log, train_loss, _, _, train_acc, val_loss, _, _, val_acc = self.train_epoch(train_loader, val_loader, epoch)

            if self.k_fold is None:
                checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + "/current_checkpoints.pth"
                self.save_checkpoint(self.model, checkpoint_path)
            else:
                checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + f"/current_checkpoints_fold_{self.k_fold}.pth"
                self.save_checkpoint(self.model, checkpoint_path)
            if val_loader:
                if self.evaluate_strategy == 'epoch':
                    val_loss_log, val_loss, _, _, val_acc = self.evaluate(val_loader, print_stats=self.cfg['training']['print_stats'], epoch=epoch)
                    self.val_losses.append(val_loss / len(val_loader))
                    self.val_accs.append(val_acc)
                
                if self.evaluate_strategy == 'epoch':
                    self.early_stopping(val_loss=val_loss_log['classification_loss'], val_acc=val_acc, model=self.model)

            if epoch % cfg['training']['log_freq'] == 0:
                print(f"[{epoch + 1}] TRAIN  loss: {train_loss / len(train_loader)} acc: {train_acc}")
                self.logging.info(f"[{epoch + 1}] TRAIN  loss: {train_loss / len(train_loader)} acc: {train_acc}")
                self.logging.info(f"[{epoch + 1}] TRAIN  loss dict: {str(train_loss_log)}")

                if val_loader:
                    print(f"[{epoch + 1}] VALIDATION loss: {val_loss / len(val_loader)} VALIDATION acc: {val_acc}")
                    self.logging.info(f"[{epoch + 1}] VALIDATION loss: {val_loss / len(val_loader)} VALIDATION acc: {val_acc}")
                    self.logging.info(f"[{epoch + 1}] VALIDATION loss dict: {str(val_loss_log)}")

                print("")
                self.logging.info("")

            self.lr_progress.append(self.optimizer.param_groups[0]["lr"])

            if self.is_early_stopping and self.early_stopping.early_stop:
                print("\n\n***Stop training***\n\n")
                self.logging.info("\n\n***Stop training***\n\n")
                break

            if self.k_fold is None:
                if val_loader is not None:
                    self.wandb.log({
                        "Loss": self.wandb.plot.line_series(
                            xs=range(len(self.train_losses)),
                            ys=[self.train_losses, self.val_losses],
                            keys=["Train Loss", "Val Loss"],
                            title="Loss",
                            xname="x epochs"
                        ),
                        "Accuracy": self.wandb.plot.line_series(
                            xs=range(len(self.train_accs)),
                            ys=[self.train_accs, self.val_accs],
                            keys=["Train Accuracy", "Valiation Accuracy"],
                            title="Accuracy",
                            xname="x epochs"
                        ),
                    })
                else:
                    self.wandb.log({
                        "Loss": self.wandb.plot.line_series(
                            xs=range(len(self.train_losses)),
                            ys=[self.train_losses, self.val_losses],
                            keys=["Train Loss", "Val Loss"],
                            title="Loss",
                            xname="x epochs"
                        )})
            else:
                self.wandb.log({
                    f"Loss Fold {self.k_fold}": self.wandb.plot.line_series(
                        xs=range(len(self.train_losses)),
                        ys=[self.train_losses, self.val_losses],
                        keys=[f"Train Loss Fold {self.k_fold}", f"Val Loss Fold {self.k_fold}"],
                        title=f"Loss Fold {self.k_fold}",
                        xname="x epochs"
                    ),
                    f"Accuracy Fold {self.k_fold}": self.wandb.plot.line_series(
                        xs=range(len(self.train_accs)),
                        ys=[self.train_accs, self.val_accs],
                        keys=[f"Train Accuracy Fold {self.k_fold}", f"Valiation Accuracy Fold {self.k_fold}"],
                        title=f"Accuracy Fold {self.k_fold}",
                        xname="x epochs"
                    ),
                })
        
        # MARK: TESTING

        print("\nTesting checkpointed models starting...\n")
        self.logging.info("\nTesting checkpointed models starting...\n")
        
        if test_loader:
            if self.k_fold is None:
                loss_checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + "/best_checkpoints_loss.pth"
                acc_checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + "/best_checkpoints_acc.pth"
            else:
                loss_checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{self.k_fold}_loss.pth"
                acc_checkpoint_path = f"checkpoints/{cfg['data']['model_name']}/" + cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{self.k_fold}_acc.pth"
            
            print("\nEvaluating checkpoint with best validation loss...")
            self.logging.info("\nEvaluating checkpoint with best validation loss...\n")
            self.model.load_state_dict(torch.load(loss_checkpoint_path))
            _, _, _, _, eval_acc_loss = self.evaluate(test_loader, print_stats=True, epoch=0)
            print("\nTesting accuracy (Best Loss Checkpoint):", eval_acc_loss)
            self.logging.info(f"\nTesting accuracy (Best Loss Checkpoint): {eval_acc_loss}")

            print("\nEvaluating checkpoint with best validation accuracy...")
            self.logging.info("\nEvaluating checkpoint with best validation accuracy...\n")
            self.model.load_state_dict(torch.load(acc_checkpoint_path))
            _, _, _, _, eval_acc_acc = self.evaluate(test_loader, print_stats=True, epoch=0)
            print("\nTesting accuracy (Best Acc Checkpoint):", eval_acc_acc)
            self.test_accuracy = eval_acc_acc  # Bạn có thể lưu trữ hoặc xử lý theo nhu cầu
            self.logging.info(f"\nTesting accuracy (Best Acc Checkpoint): {eval_acc_acc}")
        
        if self.k_fold is None:
            self.wandb.run.finish()
            
    def train_epoch(self,dataloader,val_loader = None,epoch = None):
        self.model.train()
        pred_correct, pred_all = 0, 0
        running_loss = 0.0
        loss_log = None
        n_step_loss_log = None
        self.optimizer.zero_grad()
        n_step_loss = 0
        val_loss = 0
        val_acc = 0
        val_pred_correct = 0
        val_pred_all = 0
        
        for idx, data in enumerate(tqdm(dataloader)):
            
            inputs, labels = data
            inputs = {
                key: (values.to(self.device, non_blocking=True) 
                    if isinstance(values, torch.Tensor) else values)
                for key, values in inputs.items()
            }
            labels = labels.to(self.device,non_blocking=True)
            if self.cfg['training']['mixup']:
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, labels, self.cfg['training']['alpha_mixup'], self.device)
            

            outputs = self.model(**inputs)
            if self.cfg['training']['criterion'] == "OLM_Loss":
                loss,loss_dict,logitsnorm_loss = self.criterion(**outputs, labels=labels,iteration = epoch)
                logitsnorm_loss = logitsnorm_loss / self.num_accumulation_steps
                logitsnorm_loss.backward()
            else:
                if self.cfg['training']['mixup']:
                    if outputs['logits'] is not None:
                    # Tính loss phân loại mixup dựa trên logits
                        logits = outputs['logits']
                        loss_cls, loss_dict_cls = self.criterion.classification_loss_mixup(
                            logits=logits,
                            labels_a=targets_a,
                            labels_b=targets_b,
                            lam=lam,
                            epoch=epoch
                        )
                    else:
                        loss_cls = 0
                        loss_dict_cls = {}
                    
                    aux_outputs = outputs.copy()
                    aux_outputs['logits'] = None
                    loss_aux, loss_dict_aux = self.criterion(**aux_outputs, labels=labels, epoch=epoch)
                    
                    loss = loss_cls + loss_aux
                    loss_dict = {**loss_dict_cls, **loss_dict_aux}
                else:
                    loss, loss_dict = self.criterion(**outputs, labels=labels, epoch=epoch)


            if loss_log is None:
                loss_log = loss_dict
            else:
                loss_log = {key:value + loss_dict[key] for key,value in loss_log.items()}
            
            if n_step_loss_log is None:
                n_step_loss_log = loss_dict
            else:
                n_step_loss_log = {key:value + loss_dict[key] for key,value in n_step_loss_log.items()}
            
            running_loss += loss.item()
            n_step_loss += loss.item()
            
            loss = loss / self.num_accumulation_steps
            loss.backward()
            
            if self.log_train_step and (idx+1) % self.log_steps == 0 :
                n_step_loss_log = {key:value/self.log_steps for key,value in n_step_loss_log.items()}
                self.logging.info(f"Step[{idx+1}/{len(dataloader)}]: training loss : {n_step_loss/self.log_steps} TRAIN  loss dict:  {str(n_step_loss_log)}")
                
                n_step_loss = 0
                n_step_loss_log = None
                
            if ((idx + 1) % self.num_accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                # clip grad
                if self.gradient_clip_val != 0:
                    torch.nn.utils.clip_grad_norm_(chain(self.model.parameters(), self.criterion.parameters()), max_norm=self.gradient_clip_val)
                # Update Optimizer
                self.optimizer.step()
                if self.cfg['training']['criterion'] == "OLM_Loss":
                    self.criterion.optimizer.step()  
                    self.criterion.optimizer.zero_grad()  
                self.optimizer.zero_grad()  
                
            if (idx+1) % self.evaluate_step == 0 and self.evaluate_strategy == 'step':
                if self.is_early_stopping and self.early_stopping.early_stop:
                    continue
                val_loss,val_pred_correct, val_pred_all, val_acc = self.evaluate(val_loader,print_stats=self.cfg['training']['print_stats'])
                self.early_stopping(val_loss=val_loss / len(val_loader),model = self.model)
                self.logging.info(f"Step[{idx+1}/{len(dataloader)}]: Evalutation: Val loss: {val_loss/len(val_loader)} ----- Val accuracy: {val_acc}")
                self.model.train()
                self.val_losses.append(val_loss / len(val_loader))
                self.val_accs.append(val_acc)
                self.train_losses.append(running_loss / (idx+1))
            # Statistics
            if outputs['logits'] is not None:
                logits = outputs['logits']
                pred_correct += (logits.argmax(dim = -1) == labels).sum().item()
                    
                pred_all += labels.shape[0]
        if outputs['logits'] is None:
                pred_correct = 0
                pred_all = 1   

        loss_log = {key:value/len(dataloader) for key,value in loss_log.items()}
        
        if self.evaluate_strategy == 'step':
            return loss_log,running_loss, pred_correct, pred_all, (pred_correct / pred_all),val_loss,val_pred_correct, val_pred_all, val_acc
        
        return loss_log,running_loss, pred_correct, pred_all, (pred_correct / pred_all)
    
    def train_epoch_maskfeat(self, dataloader, epoch): 
        self.model.train()
        running_loss = 0.0
        loss_log = {}
        n_step_loss = 0.0

        multi_loss_list = []  

        # Reset gradients
        self.optimizer.zero_grad()

        for idx, data in enumerate(tqdm(dataloader)):
            inputs = self.recursive_to_device(data, self.device)
            outputs = self.model(**inputs)
            loss = 0.0

            if 'preds' in outputs and 'labels' in outputs:
                preds = outputs['preds']
                labels = outputs['labels']
                loss_sum, multi_loss = self.criterion(preds, labels)
                loss += loss_sum
                multi_loss_list.append([loss_item.item() for loss_item in multi_loss])
            else:
                raise ValueError("Output của model không chứa các key dự kiến cho MaskFeat.")

            running_loss += loss.item()
            n_step_loss += loss.item()

            loss = loss / self.num_accumulation_steps
            loss.backward()

            if self.log_train_step and (idx+1) % self.log_steps == 0 :
                self.logging.info(f"Step[{idx+1}/{len(dataloader)}]: training loss : {n_step_loss/self.log_steps}")
                n_step_loss = 0

            if ((idx + 1) % self.num_accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                if self.gradient_clip_val != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if multi_loss_list:
            avg_multi_loss = [np.mean([batch_loss[i] for batch_loss in multi_loss_list])
                            for i in range(len(multi_loss_list[0]))]
            loss_log["multi_loss"] = avg_multi_loss

        loss_log["loss"] = running_loss / len(dataloader)

        if epoch > 59 and (epoch + 1) % 20 == 0:
            if self.k_fold is None:
                checkpoint_path = f"checkpoints/{self.cfg['data']['model_name']}/{self.cfg['training']['experiment_name']}/checkpoint_epoch_{epoch+1}.pth"
            else:
                checkpoint_path = f"checkpoints/{self.cfg['data']['model_name']}/{self.cfg['training']['experiment_name']}/checkpoint_epoch_{epoch+1}_fold_{self.k_fold}.pth"
            self.save_checkpoint(self.model, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

        return loss_log, running_loss, None, None, None

    def evaluate(self, dataloader, print_stats=True, epoch=None):
        self.model.eval()
        loss_log = None
        pred_correct, pred_all = 0, 0
        running_loss = 0.0
        pred = []
        gr_th = []
        stats = {i: [0, 0] for i in range(self.cfg['model']['num_classes'])}
        results = pd.DataFrame(columns=['label', 'prediction'])

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(**inputs)

                if self.cfg['training']['criterion'] == "OLM_Loss":
                    loss, loss_dict, logitsnorm_loss = self.criterion(**outputs, labels=labels, iteration=i)
                else:
                    loss, loss_dict = self.criterion(**outputs, labels=labels, epoch=epoch)

                if loss_log is None:
                    loss_log = loss_dict
                else:
                    loss_log = {key: value + loss_dict[key] for key, value in loss_log.items()}
                running_loss += loss.item()

                if outputs['logits'] is not None:
                    logits = outputs['logits']
                    prediction = logits.argmax(dim=-1)
                    pred_all += labels.shape[0]
                    pred_correct += (prediction == labels).sum().item()

                    # Add to DataFrame
                    batch_results = pd.DataFrame({
                        'label': labels.cpu().numpy(),
                        'prediction': prediction.cpu().numpy()
                    })
                    results = pd.concat([results, batch_results], ignore_index=True)

                    for idx in range(labels.shape[0]):
                        if labels[idx].item() == prediction[idx]:
                            stats[labels[idx].item()][0] += 1
                        stats[labels[idx].item()][1] += 1
                    
                    pred.extend(prediction.cpu().numpy().tolist())
                    gr_th.extend(labels.cpu().numpy().tolist())
                    
            if print_stats and outputs['logits'] is not None:
                stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
                # print("Label accuracies statistics:")
                # print(str(stats) + "\n")
                # self.logging.info("Label accuracies statistics:")
                # self.logging.info(str(stats) + "\n")

            if outputs['logits'] is None:
                pred_correct = 0
                pred_all = 1
        # Save results to CSV after loop
        results.to_csv('UsimKD_prediction.csv', index=False)

        # Existing logic to return loss and accuracy
        loss_log = {key: value / len(dataloader) for key, value in loss_log.items()}
        return loss_log, running_loss, pred_correct, pred_all, (pred_correct / pred_all)

    def evaluate_top_k(self, dataloader):
        pred_correct, pred_all = 0, 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True).reshape(-1,)

                outputs = self.model(**inputs)
                logits = outputs['logits'] 

                top_k_predictions = torch.topk(logits, self.top_k).indices.tolist()

                for idx in range(labels.shape[0]):
                    if labels[idx].item() in top_k_predictions[idx]:
                        pred_correct += 1

                pred_all += labels.shape[0]

        return pred_correct, pred_all, (pred_correct / pred_all)
    
    def evaluate_per_class(self, dataloader):
        class_correct = {i: 0 for i in range(self.cfg['model']['num_classes'])}
        class_total = {i: 0 for i in range(self.cfg['model']['num_classes'])}

        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(**inputs)
                logits = outputs['logits']
                predictions = logits.argmax(dim=-1)

                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        per_class_accuracy = [class_correct[cls] / class_total[cls] if class_total[cls] != 0 else 0 for cls in class_total]
        average_accuracy = sum(per_class_accuracy) / len(per_class_accuracy)
        return average_accuracy


    def evaluate_top_k_per_class(self, dataloader):
        class_correct = {i: 0 for i in range(self.cfg['model']['num_classes'])}
        class_total = {i: 0 for i in range(self.cfg['model']['num_classes'])}

        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True).reshape(-1,)

                outputs = self.model(**inputs)
                logits = outputs['logits']
                top_k_preds = torch.topk(logits, self.top_k).indices

                for idx, label in enumerate(labels):
                    if label.item() in top_k_preds[idx]:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        per_class_top_k_accuracy = [class_correct[cls] / class_total[cls] if class_total[cls] != 0 else 0 for cls in class_total]
        average_top_k_accuracy = sum(per_class_top_k_accuracy) / len(per_class_top_k_accuracy)
        return average_top_k_accuracy

    
