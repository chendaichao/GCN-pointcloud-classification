import torch
import numpy as np
import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, 
                 save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"net":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

def evaluate_model(data_iter, net, loss, device = device):
    acc_sum, loss_sum, n = 0.0, 0.0, 0
    net = net.to(device)
    with torch.no_grad():
        net.eval() 
        for X, y in data_iter:
            X = X.to(device)
            y_hat = net(X).to(device)
            y = y.to(device)
            loss_sum += loss(y_hat, y).cpu().item() * y.shape[0]
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
        net.train() 
    return loss_sum / n, acc_sum / n

def train_model(train_iter, valid_iter, net, loss, optimizer, lamb = 0.001, max_epochs = 100, adjust_lr = None, 
                early_stop = None, device = device):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(max_epochs):
        
        train_loss, train_acc, n, start = 0.0, 0.0, 0, time.time()
        for X, y in tqdm(train_iter, ncols = 50):
            X = X.to(device)
            y_hat, L_reg = net(X)
            y = y.to(device)
            
            l = loss(y_hat, y).to(device)
            optimizer.zero_grad()
            (l + lamb * L_reg).backward()
            optimizer.step()

            train_acc += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            train_loss += l.cpu().item() * y.shape[0]
            n += y.shape[0]
            
        valid_loss, valid_acc = evaluate_model(valid_iter, net, loss, device)
        train_loss /= n
        train_acc /= n
        
        print('epoch %d, train loss %.4f (acc %.6f), valid loss %.4f (acc %.6f), time %.1f sec'
              % (epoch, train_loss, train_acc, valid_loss, valid_acc, time.time() - start))
        
        if (adjust_lr):
            adjust_lr(optimizer)
        
        if (early_stop):
            if (early_stop(valid_loss, net, optimizer)):
                break
    
    if (early_stop):
        checkpoint = torch.load(early_stop.save_name)
        net.load_state_dict(checkpoint["net"])