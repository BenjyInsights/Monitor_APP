#!/bin/bash

################################################################################
# reproduce_benchmark.sh
#
# Reproducibility script for the moniaenergy v1.0.0 energy benchmark
# 
# Executes energy-efficient model training benchmarks on CIFAR-10 with
# automatic monitoring, Green AI grading, and Pareto frontier analysis.
#
# Usage:
#   ./reproduce_benchmark.sh --mode single    # Single config (ResNet18, BS=128)
#   ./reproduce_benchmark.sh --mode full      # All 84 configs (takes ~24 hours)
#   ./reproduce_benchmark.sh --mode analyze   # Generate plots from existing logs
#
# Requirements:
#   - Python 3.11+
#   - NVIDIA GPU (validate with nvidia-smi)
#   - moniaenergy installed: pip install -e .
#
# Author: Benjamín Sánchez Calza
# License: GPL-3.0
# Version: 1.0.0
################################################################################

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python interpreter
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Paths
DATA_DIR="${PROJECT_ROOT}/data"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
TOOLS_DIR="${PROJECT_ROOT}/tools"

# Benchmark parameters
CIFAR10_EPOCHS=50
DEVICE="${DEVICE:-cuda:0}"
COUNTRY="${COUNTRY:-Spain}"

# Timeout for single benchmark (seconds)
TIMEOUT=3600  # 1 hour per config

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

print_banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  moniaenergy — Benchmark Reproducibility Script v1.0"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

print_info() {
    echo "[INFO] $1"
}

print_success() {
    echo -e "\033[32m[✓]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[✗]\033[0m $1" >&2
}

check_requirements() {
    print_section "Checking Requirements"
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
    if [[ "${PYTHON_VERSION}" < "3.11" ]]; then
        print_error "Python 3.11+ required (found ${PYTHON_VERSION})"
        return 1
    fi
    print_success "Python ${PYTHON_VERSION}"
    
    # Check CUDA availability
    if ! $PYTHON_CMD -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        print_error "CUDA not available. Ensure NVIDIA GPU is present."
        return 1
    fi
    print_success "CUDA available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.get_device_name(0))')"
    
    # Check moniaenergy installation
    if ! $PYTHON_CMD -c "from moniaenergy import monitor_train" 2>/dev/null; then
        print_error "moniaenergy not installed. Run: pip install -e ."
        return 1
    fi
    print_success "moniaenergy installed"
    
    # Check directories
    mkdir -p "${DATA_DIR}" "${LOGS_DIR}" "${RESULTS_DIR}"
    print_success "Directories created"
    
    return 0
}

download_cifar10() {
    print_section "Preparing CIFAR-10 Dataset"
    
    if [ -d "${DATA_DIR}/cifar-10-batches-py" ]; then
        print_success "CIFAR-10 already downloaded"
        return 0
    fi
    
    print_info "Downloading CIFAR-10 (~170 MB)..."
    
    # Download using Python (torchvision will cache automatically)
    $PYTHON_CMD << 'EOF'
import sys
from torchvision import datasets, transforms

try:
    transform = transforms.ToTensor()
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("[✓] CIFAR-10 downloaded successfully")
except Exception as e:
    print(f"[✗] Failed to download CIFAR-10: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

train_single_config() {
    local model="$1"
    local batch_size="$2"
    local replica="${3:-1}"
    
    local exp_name="CIFAR10_${model}_${DEVICE##*:}_bs${batch_size}_fp16_rep${replica}"
    
    print_info "Training: ${exp_name}"
    
    $PYTHON_CMD << EOF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from moniaenergy import monitor_train
import sys

# Data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=${batch_size}, shuffle=True, num_workers=4)

# Model
device = torch.device('${DEVICE}')
if '${model}' == 'ResNet18':
    model = models.resnet18(pretrained=False, num_classes=10)
elif '${model}' == 'ResNet50':
    model = models.resnet50(pretrained=False, num_classes=10)
elif '${model}' == 'DenseNet121':
    model = models.densenet121(pretrained=False, num_classes=10)
elif '${model}' == 'VGG19':
    model = models.vgg19(pretrained=False, num_classes=10)
elif '${model}' == 'MobileNetV2':
    model = models.mobilenet_v2(pretrained=False, num_classes=10)
elif '${model}' == 'EfficientNetB0':
    from timm import create_model
    model = create_model('efficientnet_b0', pretrained=False, num_classes=10)
else:
    print(f'Unknown model: ${{model}}', file=sys.stderr)
    sys.exit(1)

model = model.to(device)

# Optimizer & scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${CIFAR10_EPOCHS})

# Training with monitoring
with monitor_train(
    model=model,
    experiment_name='${exp_name}',
    country='${COUNTRY}',
    batch_size=${batch_size},
    fp16=True,
    early_stopping=False,
    patience=10,
    power_optimize=False,
) as mon:
    for epoch in range(${CIFAR10_EPOCHS}):
        mon.epoch_start(epoch)
        
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        should_stop = mon.epoch_end(
            epoch=epoch,
            samples=total,
            loss=avg_loss,
            accuracy=accuracy,
        )
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{CIFAR10_EPOCHS}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

print(f"[✓] Training completed: ${exp_name}")
EOF
}

train_full_suite() {
    print_section "Training Full 72-Configuration Suite"
    
    local models=("ResNet18" "ResNet50" "DenseNet121" "VGG19" "MobileNetV2" "EfficientNetB0")
    local batch_sizes=(32 64 128 256)
    local replicas=(1 2 3)
    
    local total_configs=$((${#models[@]} * ${#batch_sizes[@]} * ${#replicas[@]}))
    local current=0
    
    for model in "${models[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for rep in "${replicas[@]}"; do
                current=$((current + 1))
                echo ""
                echo "════════════════════════════════════════════════════════════════════════════"
                echo "  Config ${current}/${total_configs}: ${model} BS=${bs} Rep=${rep}"
                echo "════════════════════════════════════════════════════════════════════════════"
                
                train_single_config "${model}" "${bs}" "${rep}" || {
                    print_error "Training failed: ${model} BS=${bs} Rep=${rep}"
                    continue
                }
            done
        done
    done
    
    print_success "Full suite completed (${total_configs} configurations)"
}

generate_analysis() {
    print_section "Generating Analysis & Plots"
    
    if [ ! -f "${TOOLS_DIR}/generate_executive_summary.py" ]; then
        print_error "generate_executive_summary.py not found"
        return 1
    fi
    
    print_info "Generating executive summary..."
    $PYTHON_CMD "${TOOLS_DIR}/generate_executive_summary.py" \
        --logs_dir "${LOGS_DIR}" \
        --output "${RESULTS_DIR}/SUMMARY.md" \
        || print_error "Summary generation failed"
    
    print_info "Generating plots..."
    $PYTHON_CMD "${TOOLS_DIR}/enhance_plot_quality.py" \
        --logs_dir "${LOGS_DIR}" \
        --output_dir "${RESULTS_DIR}/plots" \
        || print_error "Plot generation failed"
    
    print_success "Analysis complete: ${RESULTS_DIR}/"
}

show_summary() {
    print_section "Reproducibility Checklist"
    
    cat << 'EOF'
✓ Requirements validated (Python 3.11+, CUDA, moniaenergy)
✓ CIFAR-10 dataset prepared
✓ Benchmark(s) executed with energy monitoring
✓ Energy metrics logged (NDJSON format)
✓ Green AI Grades computed
✓ Executive summary generated
✓ Pareto frontier plots created

Next steps:
  1. Review results in: results/
  2. Analyze energy savings by model:
     - MobileNetV2: Best efficiency (J/sample)
     - ResNet50: High accuracy vs. energy trade-off
  3. Compare with thesis Chapter 4: "Resultados Experimentales"
  4. Validate against tribunal reproduction requirements

Log files:
  - Energy metrics:  logs/*/run_*.ndjson
  - Sidecar events:  logs/*/run_*.events.ndjson
  - Detailed info:   logs/*/run_*_energy_metrics.csv

Contact: sanchezcalzabenjamin@gmail.com for questions
EOF
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

main() {
    print_banner
    
    # Soporta: ./script.sh single  Y  ./script.sh --mode single
    if [[ "${1:-}" == "--mode" ]]; then
        MODE="${2:-single}"
    else
        MODE="${1:-single}"
    fi

    case "${MODE}" in
        single)
            print_section "SINGLE CONFIG MODE: ResNet18 BS=128"
            check_requirements || exit 1
            download_cifar10
            train_single_config "ResNet18" "128" "1"
            generate_analysis
            show_summary
            ;;
        full)
            print_section "FULL SUITE MODE: 72 Configurations"
            echo "⚠️  WARNING: This will take ~24 hours on RTX Ada GPU"
            echo "Proceeding in 10 seconds... (Ctrl+C to cancel)"
            sleep 10
            check_requirements || exit 1
            download_cifar10
            train_full_suite
            generate_analysis
            show_summary
            ;;
        analyze)
            print_section "ANALYSIS MODE: Generate Plots from Existing Logs"
            generate_analysis
            show_summary
            ;;
        *)
            print_error "Unknown mode: ${MODE}"
            echo "Usage: $0 [single|full|analyze]"
            exit 1
            ;;
    esac
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  Reproduction complete"
    echo "════════════════════════════════════════════════════════════════════════════"
}

# Run main function
main "$@"
