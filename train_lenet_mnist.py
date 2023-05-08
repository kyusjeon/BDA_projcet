from torchvision import transforms, datasets
from modules import lenet
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
data_path = '/mnt'
batch_size = 256
learning_rate = 1e-3

data_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

train_data = datasets.MNIST(data_path, train=True, download=True, transform=data_transform)
val_data = datasets.MNIST(data_path, train=False, download=True, transform=data_transform)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = batch_size,
    num_workers = 4,
)

model = lenet.LeNet().cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def train(epoch, log_rate, loader, model, criterion, optimizer, scheduler):
    model.train()
    step = len(loader) // log_rate
    for it, (input, target) in enumerate(tqdm(loader, leave=False)):
        it = len(loader) * epoch + it
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        wd = optimizer.param_groups[0]["weight_decay"]
        if it % step == 0:
            writer.add_scalar("train/loss", loss.item(), it)
            writer.add_scalar("train/lr", lr, it)
            writer.add_scalar("train/wd", wd, it)
        
    print(f"Epoch: [{epoch}] loss: {loss.item()}, lr: {lr}, wd: {wd}")

def validate(epoch, log_rate, loader, model, criterion):
    model.eval()
    step = len(loader) // log_rate
    
    acc1_global = 0
    acc5_global = 0
    for it, (input, target) in enumerate(tqdm(loader, leave=False)):
        it = len(loader) * epoch + it
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            
            output = model(input)
            
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1,5))
        
        if it % step == 0:
            writer.add_scalar("val/loss", loss.item(), it)
            writer.add_scalar("val/acc1", acc1.item(), it)
            writer.add_scalar("val/acc5", acc5.item(), it)
        
        acc1_global += acc1.item() / len(loader)
        acc5_global += acc5.item() / len(loader)
    print(f"Epoch: [{epoch}] loss: {loss.item()}, acc1: {acc1.item()}, acc5: {acc5.item()}")
    
    return acc1_global, acc5_global

def main(epoch, train_loader, val_loader, model, criterion, optimizer, scheduler):
    print(f"Total epoch: {epoch}")
    _acc1 = 0.
    _acc5 = 0.
    validate(-1, 1, val_loader, model, criterion)
    for it in range(epoch):
        train(it, 10, train_loader, model,criterion, optimizer, scheduler)
        acc1, acc5 = validate(it, 10, val_loader, model, criterion)
        if it%1 == 0:
            torch.save(model.state_dict(), f'lenet_mnist_{it}.pth')
            print(f"Save ckpt file: lenet_mnist_{it}.pth")
        
        if acc1 >= _acc1:
            _acc1 = acc1
            _acc5 = acc5
            
main(100, train_loader, val_loader, model, criterion, optimizer, scheduler)