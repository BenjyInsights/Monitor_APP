'''Train CIFAR10 with PyTorch.
NOTE: This script requires the pytorch-cifar model definitions.
      Clone https://github.com/kuangliu/pytorch-cifar and copy the
      models/ folder into this directory before running.
'''
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))  # moniaenergy
sys.path.insert(0, _THIS_DIR)                        # models, utils

from models import *  # requires models/ folder from pytorch-cifar repo
from utils import progress_bar
from moniaenergy import monitor_train, LayerEnergyProfiler, compute_energy_metrics, TrainingDisplay


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr',         default=0.1,      type=float, help='learning rate')
parser.add_argument('--model',      default='VGG19',  type=str,
                    help='Model: VGG19, ResNet18, MobileNetV2, DenseNet121, ...')
parser.add_argument('--epochs',     default=50,       type=int,   help='number of epochs')
parser.add_argument('--batch-size', default=128,      type=int,   help='training batch size')
parser.add_argument('--device',     default='cuda:0', type=str,
                    help='Device to use: cuda:0, cuda:1, cpu')
parser.add_argument('--fp16',           action='store_true',       help='mixed precision (torch.cuda.amp)')
parser.add_argument('--early-stopping',       action='store_true',      help='enable EnergyEarlyStopping')
parser.add_argument('--power-optimize',       action='store_true',      help='enable Zeus-style GPU power optimization')
parser.add_argument('--time-budget',          default=0.10, type=float, help='tolerated slowdown for power auth')
parser.add_argument('--min-efficiency',       default=None, type=float, help='absolute ΔAcc/J threshold (auto-calibrates if omitted)')
parser.add_argument('--min-efficiency-ratio', default=0.05, type=float, help='threshold = ratio × first-epoch efficiency (default: 0.05)')
parser.add_argument('--patience',             default=3,    type=int,   help='consecutive inefficient epochs before stopping')
parser.add_argument('--resume', '-r',   action='store_true',       help='resume from checkpoint')
parser.add_argument('--display',        action='store_true',       help='rich live terminal dashboard (disables progress_bar)')
parser.add_argument('--rep',            default=1,    type=int,   help='repetition index (for statistical replication; encoded in run_name)')
parser.add_argument('--quiet',          action='store_true',       help='suppress per-batch progress bar (clean logs for unattended runs)')
args = parser.parse_args()

# Optimization mode tag derived from flags (Control / Zeus / Full) — used in run_name
# for one-directory-per-run traceability across repetitions.
if args.power_optimize and args.early_stopping:
    _mode_tag = 'full'      # Full_Optimized (power capping + EES)
elif args.power_optimize:
    _mode_tag = 'zeus'      # Zeus_Only (power capping)
else:
    _mode_tag = 'control'   # Control (baseline)

device     = args.device
_amp_dtype = device.split(':')[0]  # 'cuda' or 'cpu'

if _amp_dtype == 'cuda' and not torch.cuda.is_available():
    print('CUDA not available, falling back to cpu')
    device     = 'cpu'
    _amp_dtype = 'cpu'

print('=' * 78)
print(f'[RUN] Modelo={args.model}  Batch={args.batch_size}  Modo={_mode_tag.upper()}  '
      f'Rep={args.rep}  Épocas={args.epochs}')
print(f'[RUN] Device={device}  fp16={args.fp16}  '
      f'power_optimize={args.power_optimize}  early_stopping={args.early_stopping}')
print(f'[RUN] run_name=CIFAR10_{args.model}_{device.replace(":", "")}_bs{args.batch_size}'
      f'{"_fp16" if args.fp16 else "_fp32"}_{_mode_tag}_rep{args.rep}')
print('=' * 78)

if _amp_dtype == 'cuda':
    cudnn.benchmark = True

# Optional rich display — set to a TrainingDisplay instance when --display is active
_display: "TrainingDisplay | None" = None

# ── Data ──────────────────────────────────────────────────────────────────────
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_dir    = os.path.join(_THIS_DIR, 'data')
trainset    = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform_train)
testset     = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ── Model ─────────────────────────────────────────────────────────────────────
print(f'==> Building model: {args.model}')
_model_registry = {
    'VGG19':          lambda: VGG('VGG19'),
    'ResNet18':       ResNet18,
    'ResNet50':       ResNet50,
    'MobileNetV2':    MobileNetV2,
    'DenseNet121':    DenseNet121,
    'EfficientNetB0': EfficientNetB0,
    'ViT':            ViT,
}
net = _model_registry.get(args.model, lambda: VGG('VGG19'))()

profiler = LayerEnergyProfiler(net, device=device)
net      = net.to(device)

# ── Checkpoint / resume ───────────────────────────────────────────────────────
ckpt_dir  = os.path.join(_THIS_DIR, 'checkpoint')
ckpt_path = os.path.join(ckpt_dir, f'{args.model.lower()}_ckpt.pth')
best_acc  = 0
start_epoch = 0

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(ckpt_path), f'Error: checkpoint not found at {ckpt_path}'
    checkpoint_data = torch.load(ckpt_path)
    net.load_state_dict(checkpoint_data['net'])
    best_acc    = checkpoint_data['acc']
    start_epoch = checkpoint_data['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scaler    = torch.amp.GradScaler(_amp_dtype, enabled=(args.fp16 and _amp_dtype == 'cuda'))


def train(epoch) -> tuple:
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=_amp_dtype, enabled=args.fp16):
            outputs = net(inputs)
            loss    = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if _display is not None:
            _display.update_batch(
                "TRAIN", batch_idx, len(trainloader),
                train_loss / (batch_idx + 1), correct / total,
            )
        elif not args.quiet:
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss / len(trainloader), correct / total


def test(epoch) -> tuple:
    global best_acc
    net.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=_amp_dtype, enabled=args.fp16):
                outputs = net(inputs)
                loss    = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if _display is not None:
                _display.update_batch(
                    "TEST", batch_idx, len(testloader),
                    test_loss / (batch_idx + 1), correct / total,
                )
            elif not args.quiet:
                progress_bar(batch_idx, len(testloader),
                             'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving checkpoint..')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({'net': net.state_dict(), 'acc': acc, 'epoch': epoch}, ckpt_path)
        best_acc = acc

    return test_loss / len(testloader), correct / total


# ── Log paths ─────────────────────────────────────────────────────────────────
# Las rutas base son administradas internamente por la Fachada `monitor_train`.
_device_tag = device.replace(':', '')
_prec_tag   = '_fp16' if args.fp16 else '_fp32'
log_ndjson  = None

# GPU index — derived from --device regardless of --display
_gpu_idx = int(device.split(":")[-1]) if "cuda" in device else 0

# ── Optional rich display ──────────────────────────────────────────────────────
if args.display:
    _display = TrainingDisplay(args.model, device, args.fp16, args.epochs, gpu_index=_gpu_idx)
    _display.__enter__()

# ── Training loop inside Facade ───────────────────────────────────────────────
try:
    with monitor_train(
        model=net,
        run_name=f"CIFAR10_{args.model}_{_device_tag}_bs{args.batch_size}{_prec_tag}_{_mode_tag}_rep{args.rep}",
        log_dir=os.path.join(_PROJECT_ROOT, "logs"),
        power_optimize=args.power_optimize,
        time_budget_pct=args.time_budget,
        early_stopping=args.early_stopping,
        fp16=args.fp16,
        patience=args.patience,
        min_efficiency=args.min_efficiency,
        min_efficiency_ratio=args.min_efficiency_ratio,
        gpu_index=_gpu_idx,
        batch_size=args.batch_size,
    ) as session:
        log_ndjson = session._log_path
        
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if _display is None:
                print(f'\nEpoch: {epoch}')
            
            session.epoch_start(epoch)

            train_loss, train_acc = train(epoch)
            test_loss,  test_acc  = test(epoch)
            scheduler.step()

            _ees_stop = session.epoch_end(
                epoch=epoch,
                samples=len(trainset),
                loss=train_loss,
                accuracy=train_acc,
            )

            if _display is None:
                print(f'[EPOCH {epoch}] train_loss={train_loss:.4f} '
                      f'train_acc={train_acc:.4f} test_acc={test_acc:.4f}'
                      + ('  [EES -> STOP]' if _ees_stop else ''), flush=True)

            if _display is not None:
                _em = compute_energy_metrics(log_ndjson)
                if not _em.empty:
                    _row     = _em[_em["epoch"] == epoch]
                    _co2_col = next((c for c in _em.columns if "europe" in c), None)
                else:
                    _row     = None
                    _co2_col = None
                _has_row = _row is not None and not _row.empty
                _display.update_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    energy_j  = float(_row["total_energy_j"].iloc[0])      if _has_row else None,
                    eps_j     = float(_row["energy_per_sample_j"].iloc[0]) if _has_row else None,
                    edp       = float(_row["edp"].iloc[0])                  if _has_row else None,
                    co2_eu    = float(_row[_co2_col].iloc[0]) if (_has_row and _co2_col) else None,
                    ees_status= f"patience {session._ees._bad_epochs}/{session._ees._patience}" if session._ees else None,
                    energy_grade = str(_row["energy_grade"].iloc[0]) if (_has_row and "energy_grade" in _row.columns and _row["energy_grade"].iloc[0] is not None) else None,
                )

            if _ees_stop:
                break
finally:
    if _display is not None:
        _display.__exit__(None, None, None)

# ── Layer profiler summary ─────────────────────────────────────────────────────
layer_df = profiler.get_summary()
profiler.remove()
print("\n--- Layer Compute Time Summary (top 10) ---")
print(layer_df.head(10).to_string(index=False))

# ── Post-hoc energy analysis ───────────────────────────────────────────────────
energy_df  = compute_energy_metrics(log_ndjson)

if not energy_df.empty:
    cols     = ["epoch", "duration_s", "total_energy_j", "energy_per_sample_j", "edp"]
    co2_cols = [c for c in energy_df.columns if c.startswith("co2_")]
    print("\n--- Per-Epoch Energy Metrics ---")
    print(energy_df[cols + co2_cols].to_string(index=False))

    energy_csv = log_ndjson.replace(".ndjson", "_energy_metrics.csv")
    layer_csv  = log_ndjson.replace(".ndjson", "_layer_profile.csv")
    energy_df.to_csv(energy_csv, index=False)
    layer_df.to_csv(layer_csv, index=False)
    print(f"\nResults saved to:\n  {energy_csv}\n  {layer_csv}")
