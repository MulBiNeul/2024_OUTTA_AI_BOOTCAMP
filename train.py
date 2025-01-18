from read_dataset import *
from preprop import *
from model import segmentation_model
import torch
import torch.nn.functional as F

def train(img_list, gt_list, model, epoch, learning_rate, optimizer, criterion, data_len, device):
    running_loss = 0.0
    model.train()

    for i in range(epoch):
        for iter in range(data_len):
            optimizer.zero_grad()
            
            inputs = torch.tensor(img_list[iter]).float().to(device)
            targets = torch.tensor(gt_list[iter]).long().to(device)
            targets = F.interpolate(targets.unsqueeze(0).float(), size=(132, 132), mode='nearest').long()

            outputs = model(inputs.unsqueeze(0))
            loss = criterion(outputs, targets.squeeze(0))
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (iter % 100 == 0) & (iter != 0):
                print(f'Iteration: {iter+data_len*i}, Loss: {running_loss / (iter+1+data_len*i)}')
        torch.save(model.state_dict(), f'model_state_dict{i}.pth')