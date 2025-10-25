from __future__ import annotations
from pathlib import Path
import nbformat as nbf
from datetime import datetime
from csfl_simulator.core.utils import ROOT


def export_config_to_ipynb(config: dict, method_code: str, output_path: Path):
    nb = nbf.v4.new_notebook()
    nb.cells = []
    nb.cells.append(nbf.v4.new_markdown_cell(f"# CSFL Simulation Export\nGenerated: {datetime.now().isoformat()}"))
    nb.cells.append(nbf.v4.new_code_cell("""
# Install (adjust for CUDA/CPU):
# pip install torch torchvision numpy pandas scikit-learn matplotlib plotly seaborn nbformat nbclient pyyaml rich tqdm
# Optional (GNN): pip install torch-scatter torch-sparse torch-geometric  # choose wheels from https://data.pyg.org/whl
"""))
    nb.cells.append(nbf.v4.new_code_cell(f"CONFIG = {config!r}\nprint('Loaded CONFIG:', CONFIG)"))
    nb.cells.append(nbf.v4.new_markdown_cell("## Imports"))
    nb.cells.append(nbf.v4.new_code_cell("""
import math, json, random, copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("## Dataset & Partition"))
    nb.cells.append(nbf.v4.new_code_cell("""
name = CONFIG['dataset']
if name.lower()=='mnist':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
elif name.lower() in ['fashion-mnist','fashionmnist']:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
elif name.lower() in ['cifar-10','cifar10']:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
else:
    raise ValueError('Unsupported dataset in export')

# Partition
num_clients = int(CONFIG['total_clients'])
labels = [train_ds[i][1] for i in range(len(train_ds))]
import numpy as np
if CONFIG['partition']=='iid':
    idxs = np.arange(len(labels)); np.random.shuffle(idxs)
    mapping = {i: idxs[i::num_clients].tolist() for i in range(num_clients)}
elif CONFIG['partition']=='dirichlet':
    alpha = float(CONFIG.get('dirichlet_alpha',0.5))
    y = np.array(labels); C = int(np.max(y)+1)
    cls = [np.where(y==c)[0] for c in range(C)]
    mapping = {i: [] for i in range(num_clients)}
    for c in range(C):
        n=len(cls[c]); np.random.shuffle(cls[c])
        props = np.random.dirichlet(alpha*np.ones(num_clients))
        cuts = (np.cumsum(props)*n).astype(int)[:-1]
        parts = np.split(cls[c], cuts)
        for i,part in enumerate(parts): mapping[i].extend(part.tolist())
else:
    shards_per_client = int(CONFIG.get('shards_per_client',2))
    idxs = np.arange(len(labels)); np.random.shuffle(idxs)
    num_shards=num_clients*shards_per_client
    shard_size=len(idxs)//num_shards
    shards=[idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    mapping={i:[] for i in range(num_clients)}
    order=np.random.permutation(num_shards)
    for i,s in enumerate(order): mapping[i%num_clients].extend(shards[s].tolist())

batch_size=int(CONFIG['batch_size'])
client_loaders={}
for cid in range(num_clients):
    sub=Subset(train_ds, mapping[cid])
    client_loaders[cid]=DataLoader(sub, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
print('Partition done: clients=',num_clients)
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("## Model"))
    nb.cells.append(nbf.v4.new_code_cell("""
class CNNMnist(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,num_classes)
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2(x),2))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class LightCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(128*4*4,256)
        self.fc2=nn.Linear(256,num_classes)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

name=CONFIG['model'].lower()
num_classes = len(getattr(test_ds,'classes',[])) or 10
if name in ['cnn-mnist','cnn_mnist']:
    model = CNNMnist(num_classes)
elif name in ['lightcnn','light-cifar']:
    model = LightCIFAR(num_classes)
else:
    # Fallback to CNNMnist for MNIST, LightCIFAR otherwise
    model = CNNMnist(num_classes) if 'mnist' in CONFIG['dataset'].lower() else LightCIFAR(num_classes)

device = 'cuda' if torch.cuda.is_available() and CONFIG.get('device','auto')!='cpu' else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=float(CONFIG['lr']))
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("## Selection Method"))
    nb.cells.append(nbf.v4.new_code_cell(method_code))
    nb.cells.append(nbf.v4.new_markdown_cell("## Federated Loop"))
    nb.cells.append(nbf.v4.new_code_cell("""
K = int(CONFIG['clients_per_round'])
rounds = int(CONFIG['rounds'])
fast = bool(CONFIG.get('fast_mode', True))

history = {"selected": []}

def evaluate(m):
    m.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device), y.to(device)
            out = m(x)
            pred=out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

metrics=[]
base_acc = evaluate(model)
metrics.append({"round": -1, "accuracy": base_acc})

for rnd in range(rounds):
    # Build client info
    class C:
        pass
    clients=[]
    for cid in range(num_clients):
        c=C(); c.id=cid; c.data_size=len(client_loaders[cid].dataset); c.last_loss=0.0; c.grad_norm=0.0; c.compute_speed=1.0; c.channel_quality=1.0; c.participation_count=0
        clients.append(c)
    # Select
    ids, _, _ = select_clients(rnd, K, clients, history, random, None, device)
    history['selected'].append(ids)
    # Local train
    updates=[]; weights=[]
    for cid in ids:
        local=type(model)() if hasattr(model,'__class__') else model
        local=type(model)(*[]) if False else copy.deepcopy(model)
        local=local.to(device)
        local.load_state_dict(model.state_dict())
        local.train()
        last_loss=0.0
        for e in range(int(CONFIG['local_epochs'])):
            for bi,(x,y) in enumerate(client_loaders[cid]):
                x,y=x.to(device), y.to(device)
                opt_l=optim.SGD(local.parameters(), lr=float(CONFIG['lr']))
                opt_l.zero_grad()
                loss=criterion(local(x),y)
                loss.backward(); opt_l.step()
                last_loss=float(loss.item())
                if fast and bi>1: break
        updates.append(copy.deepcopy(local.state_dict())); weights.append(len(client_loaders[cid].dataset))
    # FedAvg
    if updates:
        new_sd=copy.deepcopy(model.state_dict())
        total=sum(weights)
        for k in new_sd.keys():
            new_sd[k]=sum( (sd[k]*(w/total) for sd,w in zip(updates,weights) ) )
        model.load_state_dict(new_sd)
    acc=evaluate(model)
    metrics.append({"round": rnd, "accuracy": acc})
    print(f"Round {rnd+1}/{rounds}: acc={acc:.4f}")

# Plot
plt.figure(figsize=(6,3))
plt.plot([m['round'] for m in metrics],[m['accuracy'] for m in metrics], marker='o')
plt.xlabel('Round'); plt.ylabel('Accuracy'); plt.title('FL Accuracy'); plt.grid(True)
plt.show()
"""))
    # Ensure output path is under project artifacts/exports when relative
    if not output_path.is_absolute():
        output_path = (ROOT / "artifacts" / "exports" / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(output_path))
    return str(output_path)
