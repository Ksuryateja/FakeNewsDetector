import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_model(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, epoch_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list, 'valid_loss_list': valid_loss_list, 'train_acc_list': train_acc_list, 'valid_acc_list': valid_acc_list,'epoch_list': epoch_list}
    
    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Metrics loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['epoch_list']

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0
    return (y_true == y_prob).sum().item() / y_true.size(0)