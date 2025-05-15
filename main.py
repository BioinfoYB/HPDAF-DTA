import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model_f_3_2 import MultiFusion_DTA
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast


def train(model, device, train_loader, optimizer,loss_fn):
    model.train()
    for batch_idx, data in tqdm(enumerate(train_loader), disable=True, total=len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(data)
            loss = loss_fn(output, data.y_s.view(-1, 1).float().to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y_s
            y_hat = model(data)
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    test_loss /= len(test_loader.dataset)
    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
        'AUC': metrics.AUC(targets, outputs)  
    }

    return evaluation




scaler = torch.amp.GradScaler()
device = torch.device("cuda")
model = MultiFusion_DTA().to(device)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters())
data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size= 64,
                           pin_memory=True,
                           # num_workers=8,
                           shuffle=True,follow_batch=['x_s','x_t'])
            for phase_name in ['train','val','test2016','test2013','BDBbind']}
time_now = datetime.now()

NUM_EPOCHS = 100
save_best_epoch=30
best_val_loss = 100000000
best_epoch = -1

start = datetime.now()
print('start at ', start)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"The {epoch}/{NUM_EPOCHS} epoch")

    train(model, device, data_loaders['train'], optimizer, loss_fn)

    for _p in ['val', 'test2016', 'test2013', 'BDBbind']:
        performance = test(model, data_loaders[_p], loss_fn, device, False)

        # Check if it's the best validation loss
        if _p == 'val' and epoch >= save_best_epoch and performance['loss'] < best_val_loss:
            best_val_loss = performance['loss']
            best_epoch = epoch
            best_performance = performance  # Save the full performance dictionary


    if best_epoch != -1:
        print(f"The best epoch is {best_epoch} epoch")
        print(f"Best validation loss at epoch {best_epoch}: {best_val_loss}")
        print(f"Other metrics at best epoch: {best_performance}")



print('train finished')
end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))