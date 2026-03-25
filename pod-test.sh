#!/bin/bash
# ═══════════════════════════════════════════════════════════
# RunPod GPU Benchmark
# Run this first on any new RunPod pod to measure actual
# GPU performance before committing to an expensive run.
# Takes ~30 seconds.
#
# Usage: bash pod-test.sh
# ═══════════════════════════════════════════════════════════

set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  RunPod GPU Benchmark"
echo "═══════════════════════════════════════════"
echo ""

# 1. GPU Identity
echo "=== GPU Identity ==="
nvidia-smi --query-gpu=name,pci.bus_id,serial,uuid --format=csv,noheader
echo ""

# 2. Clock speeds (higher = faster)
echo "=== Clock Speeds ==="
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,clocks.mem,clocks.max.mem --format=csv,noheader
echo "(graphics_clock, max_graphics, mem_clock, max_mem) in MHz"
echo ""

# 3. Memory and PCIe info
echo "=== Memory & PCIe ==="
nvidia-smi --query-gpu=memory.total,memory.free,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
echo ""

# 4. NVLink (matters for multi-GPU)
echo "=== NVLink ==="
nvidia-smi nvlink --status 2>/dev/null | head -5 || echo "NVLink info not available (single GPU or not supported)"
echo ""

# 5. CPU info (affects data loading)
echo "=== CPU ==="
lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Thread|MHz" | head -5
echo ""

# 6. Disk speed (affects data loading)
echo "=== Disk Speed ==="
dd if=/dev/zero of=/tmp/bench_test bs=1M count=1024 2>&1 | tail -1
rm -f /tmp/bench_test
echo ""

# 7. Network (affects data download)
echo "=== Network ==="
curl -s -o /dev/null -w "Download speed: %{speed_download} bytes/sec\n" https://huggingface.co/robots.txt
echo ""

# 8. ACTUAL GPU COMPUTE BENCHMARK — matrix multiply
echo "=== GPU Compute (GEMM) ==="
python3 -c "
import torch, time

device = 'cuda'
# Warmup
a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = sum(times) / len(times)
tflops = 2 * 4096**3 / avg / 1e12
print(f'GEMM 4096x4096 bf16: {avg*1000:.2f} ms  ({tflops:.1f} TFLOPS)')
print(f'Min: {min(times)*1000:.2f} ms  Max: {max(times)*1000:.2f} ms')

# Memory bandwidth test
n = 256 * 1024 * 1024  # 256M elements = 512MB
x = torch.randn(n, device=device, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    y = x + 1.0
torch.cuda.synchronize()
t1 = time.perf_counter()
bw = 10 * 2 * n * 2 / (t1 - t0) / 1e9  # read + write, 2 bytes each
print(f'Memory bandwidth: {bw:.0f} GB/s')
" 2>&1
echo ""

# 9. Multi-GPU check
echo "=== GPU Count ==="
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $GPU_COUNT"
if [ "$GPU_COUNT" -gt 1 ]; then
    echo ""
    echo "=== All GPUs ==="
    nvidia-smi --query-gpu=index,name,clocks.gr,clocks.max.gr,memory.total,memory.free --format=csv,noheader
    echo ""
    echo "=== GPU-to-GPU Bandwidth (NVLink/PCIe) ==="
    nvidia-smi topo -m 2>/dev/null || echo "Topology info not available"
fi
echo ""

# 10. Location hint
echo "=== Pod Location ==="
IP=$(curl -s ifconfig.me 2>/dev/null)
CITY=$(curl -s ipinfo.io/city 2>/dev/null)
REGION=$(curl -s ipinfo.io/region 2>/dev/null)
COUNTRY=$(curl -s ipinfo.io/country 2>/dev/null)
echo "$IP — $CITY, $REGION, $COUNTRY"
echo ""

echo "═══════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo ""
echo "  REFERENCE VALUES (good H100 SXM):"
echo "  GEMM: < 0.50 ms (> 275 TFLOPS)"
echo "  Memory BW: > 2800 GB/s"
echo "  Max GPU clock: 1980 MHz"
echo ""
echo "  If GEMM > 0.70 ms or BW < 2000 GB/s,"
echo "  consider stopping and getting a new pod."
echo "═══════════════════════════════════════════"
