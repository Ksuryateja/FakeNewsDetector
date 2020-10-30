# -*- coding: utf-8 -*-
from models import normal, attention
import dataloader
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchtext import data
from torchtext.data import BucketIterator
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm

writer = SummaryWriter()

def train(train_loader, valid_loader,
        epochs, 
        lr, 
        model,
        criterion, 
        checkpoint_path,
        best_valid_loss = float("Inf")):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list =[]
    epoch_list = []


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-5)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        with tqdm(total = len(train_loader), desc='Train', ncols=75) as pbar:
            for step, batch in enumerate(train_loader):
                labels = batch.labels
                pbar.update(len(batch))
                optimizer.zero_grad()
                labels = Variable(labels.unsqueeze(dim = 1)).to(device)

                output = model(batch)
                train_loss = criterion(output, labels)
                writer.add_scalar("Loss/train", train_loss, epoch+1)
              
                acc = 100 * utils.get_accuracy(labels.squeeze(dim=1), output.squeeze(dim=1))
                writer.add_scalar("Accuracy/train", acc, epoch+1)

                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item()
                epoch_acc += acc
                

                if step % 500 == 0:
                    tqdm.write(f'| Epoch: {epoch+1:03}/{epochs} | Train Loss: {train_loss.item():10.4f} | Learning Rate: {lr:.4f} |')

            tqdm.write(f'Evaluating on {epoch+1:,}')
            valid_loss, valid_acc = evaluate(model, valid_loader, epoch, criterion)
            
            train_loss_list.append(epoch_loss / len(train_loader))
            train_acc_list.append(epoch_acc / len(train_loader))
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            epoch_list.append(epoch)

            tqdm.write(f'Training Loss: {train_loss_list[-1]:.4f} | Training Accuracy: {train_acc_list[-1]:.2f}%')
            tqdm.write(f'Validation Loss: { valid_loss:.4f} | Validation Accuracy: {valid_acc:.2f}%')
            tqdm.write('\n')

            #checkpoint
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                utils.save_model(checkpoint_path + '/model.pt', model, optimizer, best_valid_loss)
            utils.save_metrics(checkpoint_path + '/metrics.pt', train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, epoch_list)

def evaluate(model, loader, epoch, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            labels = batch.labels
            labels = Variable(labels.unsqueeze(dim = 1)).to(device)

            output = model(batch)
            loss = criterion(output, labels)
            writer.add_scalar("Loss/test", loss, epoch+1)
            epoch_loss += loss.item()

            acc = 100 * utils.get_accuracy(labels.squeeze(dim=1), output.squeeze(dim=1))
            writer.add_scalar("Accuracy/test", acc, epoch+1)
            epoch_acc += acc

        total_loss = epoch_loss / len(loader)
        total_acc = epoch_acc / len(loader)
        return total_loss, total_acc

def test(model, loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    total_loss = 0
    total_acc = 0
    
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            labels = batch.labels
            labels = Variable(labels.unsqueeze(dim = 1)).to(device)

            output = model(batch)
            loss = criterion(output, labels)
            
            total_loss += loss.item()

            acc = 100 * utils.get_accuracy(labels.squeeze(dim=1), output.squeeze(dim=1))
            
            total_acc += acc

        total_loss = total_loss / len(loader)
        total_acc = total_acc / len(loader)
        return total_loss, total_acc


train_iter, valid_iter, test_iter, vocab_size, word_embeddings = dataloader.Embedding(batch_size = 32, model='attention')
LiarNet = attention.LiarNet(vocab_size = vocab_size, embed_weights = word_embeddings, batch_size = 32)
criterion = nn.BCEWithLogitsLoss()
#train(train_loader= train_iter, valid_loader= valid_iter, epochs= 100, lr= 3e-3, model= LiarNet,criterion = criterion, checkpoint_path='./model_checkpoint/attention')
writer.flush()
writer.close()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = attention.LiarNet(vocab_size = vocab_size, embed_weights = word_embeddings, batch_size = 32)
best_model.to(device)
optimizer = torch.optim.Adam(best_model.parameters(), lr=0.01)
utils.load_checkpoint('./model_checkpoint/attention' + '/model.pt', best_model, optimizer)
test_loss, test_acc = test(best_model, test_iter, criterion)
print(f'Test Accuracy: {test_acc:.2f}%')