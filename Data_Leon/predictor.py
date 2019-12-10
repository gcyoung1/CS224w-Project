import sqlite_process as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from os import listdir, mkdir
from os.path import join, isdir, isfile

class LinearPredictor(nn.Module):
    def __init__(self, num_inputs, num_outputs, layers):
        super().__init__()
        layers = [num_inputs, *layers, num_outputs]
        self.nets = []
        for i in range(len(layers)-1):
            self.nets.append(nn.Linear(layers[i], layers[i+1]))
            self.add_module('Linear_%d'%i, self.nets[-1])
    def forward(self, x):
        for net in self.nets[:-1]:
            x = F.relu(net(x))
        x = self.nets[-1](x)
        return x
    def predict(self, x, y, do_print=True):
        x = self.forward(x)
        predictions = torch.max(x, 1)[1]
        is_correct = predictions == y
        num_cats = x.shape[1]
        n_each = torch.tensor([torch.sum(y==i) for i in range(num_cats)])
        c_each = torch.tensor([torch.sum(is_correct[y==i]) for i in range(num_cats)])
        n_total = torch.sum(n_each)
        c_total = torch.sum(c_each)
        accs = 1.*c_each/n_each
        if do_print:
            print("Total Accuracy: %d/%d (%.4f)" % (c_total, n_total, 1.*c_total/n_total))
            print("Ave Accuracies (%.4f)" % torch.mean(accs))
            for i in range(num_cats):
                print("\tType %d Accuracy: %d/%d (%.3f)" % (i, c_each[i], n_each[i], accs[i]))
        return torch.mean(accs)

class MatrixArrayDataset(Dataset):
    def __init__(self, matrix, array):
        self.matrix = matrix
        self.array = array
    def __len__(self):
        return self.array.shape[0]
    def __getitem__(self, idx):
        return self.matrix[idx], self.array[idx]

def save(model_name, iteration, model, optimizer, save_dir='Models', loss_history=None, print_res=True):
    if not isdir(save_dir):
        mkdir(save_dir)
    if not isdir(join(save_dir, model_name)):
        mkdir(join(save_dir, model_name))
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    if type(iteration) == int:
        iteration = "%d"%iteration
    save_to = join(save_dir, model_name, "%s-%s" % (model_name, iteration))
    torch.save(state, save_to)
    if loss_history:
        torch.save(loss_history, join(save_dir, model_name, "loss"))
    if print_res:
        print('model saved to %s' % save_to)

def load(model_name, iteration, model, optimizer, save_dir='Models', map_location=None):
    if type(iteration) == int:
        iteration = "%d"%iteration
    save_to = join(save_dir, model_name, "%s-%s"%(model_name, iteration))
    if map_location is not None:
        state = torch.load(save_to, map_location=map_location)
    else:
        state = torch.load(save_to)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % save_to)

def DSET_Split(x, y, train_val_test = (.7,.2,.1)):
    if sum(train_val_test) - 1 > 1e-10:
        print("Splits must sum to 1")
        return
    split1, split2 = int(len(y) * train_val_test[0]), int(len(y) * (train_val_test[0] + train_val_test[1]))
    xtrain, ytrain = x[:split1], y[:split1]
    xval, yval = x[split1:split2], y[split1:split2]
    xtest, ytest = x[split2:], y[split2:]
    return xtrain, ytrain, xval, yval, xtest, ytest

def TrainPredictor(x, y, layers, num_epochs, train_val_test = (.7,.2,.1),
                   optim_params=None, model_name=None, save_every=10, predict_every=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)
    xtrain, ytrain, xval, yval, xtest, ytest = DSET_Split(x, y, train_val_test)
    counts = torch.bincount(ytrain)
    weights = 1. * torch.sum(counts) / counts / len(counts)
    dl = DataLoader(MatrixArrayDataset(xtrain, ytrain), batch_size=256)
    model = LinearPredictor(x.shape[1], len(counts), layers).to(device)
    default_optim_params = {
        'lr': 1e-4,
        'momentum': .9,
        'weight_decay': 1e-7,
        'nesterov': True
    }
    if optim_params is not None:
        for k, v in optim_params.items():
            default_optim_params[k] = v
    optim = torch.optim.SGD(model.parameters(), **default_optim_params)
    losses_all = []
    val_losses = []
    min_val_loss = 100
    best_epoch = -1

    def run_validation(e, best_epoch, min_val_loss):
        with torch.no_grad():
            loss = F.cross_entropy(model(xval), yval, weights).item()
            if model_name:
                if loss < min_val_loss:
                    # save best model
                    best_epoch = e
                    min_val_loss = loss
                    save(model_name, 'Best', model, optim, print_res=False)
                if e % save_every == 0:
                    save(model_name, e, model, optim)
                if e % predict_every == 0:
                    print('\nPredictions on Training Set')
                    model.predict(xtrain, ytrain)
                    print('\nPredictions on Validation Set')
                    model.predict(xval, yval)
        return loss, best_epoch, min_val_loss

    def run_train():
        losses = []
        for x_, y_ in dl:
            loss = F.cross_entropy(model(x_), y_, weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        return losses

    for e in range(num_epochs):
        # Validation
        val_loss, best_epoch, min_val_loss = run_validation(e, best_epoch, min_val_loss)
        val_losses.append(val_loss)
        # Train
        losses = run_train()
        losses_all += losses
        print("Epoch %d, train loss %.5f, val loss %.5f" % (e, torch.mean(torch.tensor(losses)), val_losses[-1]))
    val_loss, best_epoch, min_val_loss = run_validation(num_epochs, best_epoch, min_val_loss)
    print("\n\nBest epoch is %d, with val loss %.4f" % (best_epoch, min_val_loss))
    if model_name:
        with torch.no_grad():
            best_model = LinearPredictor(x.shape[1], len(counts), layers).to(device)
            _ = torch.optim.SGD(model.parameters(), **default_optim_params)
            load(model_name, 'Best', best_model, _)
            print('\nPredictions on Training Set')
            best_model.predict(xtrain, ytrain)
            print('\nPredictions on Validation Set')
            best_model.predict(xval, yval)
            print('\nPredictions on test Set')
            best_model.predict(xtest, ytest)
    return model, optim, losses_all, val_losses





