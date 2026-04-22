#!/bin/bash
# ═══════════════════════════════════════════════════════════
# RunPod GPU Benchmark
# Run this first on any new RunPod pod to measure actual
# GPU performance before committing to an expensive run.
# Takes ~30 seconds.
#
# Usage:
#   bash pod-test.sh                                 # stdout only
#   bash pod-test.sh | tee pod-$(date +%s).txt       # save human-readable
#   JSON_OUT=mygpu.json bash pod-test.sh             # also write JSON sidecar
#                                                    # (default: /tmp/pod-benchmark-<ts>.json)
#
# JSON sidecar captures pod id, datacenter, per-GPU UUID and serial,
# GEMM and memory bandwidth numbers. Use it to identify and request the
# same specific GPU again later. See "Finding a specific GPU again"
# in the README.
# ═══════════════════════════════════════════════════════════

set -uo pipefail

echo "═══════════════════════════════════════════"
echo "  RunPod GPU Benchmark"
echo "═══════════════════════════════════════════"
echo ""

# 0. RunPod pod and datacenter identifiers (captured early so they land in the log)
echo "=== RunPod Pod Identity ==="
POD_ID="${RUNPOD_POD_ID:-unknown}"
POD_HOSTNAME="${RUNPOD_POD_HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"
POD_DC_ID="${RUNPOD_DC_ID:-${RUNPOD_REGION:-unknown}}"
POD_PUBLIC_IP="${RUNPOD_PUBLIC_IP:-$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo unknown)}"
POD_CITY="$(curl -s --max-time 3 ipinfo.io/city 2>/dev/null || echo unknown)"
POD_REGION="$(curl -s --max-time 3 ipinfo.io/region 2>/dev/null || echo unknown)"
POD_COUNTRY="$(curl -s --max-time 3 ipinfo.io/country 2>/dev/null || echo unknown)"
echo "pod_id=${POD_ID}"
echo "pod_hostname=${POD_HOSTNAME}"
echo "pod_public_ip=${POD_PUBLIC_IP}"
echo "pod_dc_id=${POD_DC_ID}"
echo "pod_location=${POD_CITY}, ${POD_REGION}, ${POD_COUNTRY}"
echo ""

# 1. GPU Identity (name, PCI, serial, UUID)
echo "=== GPU Identity ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,serial,uuid --format=csv,noheader
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
GEMM_OUTPUT=$(python3 -c "
import torch, time, json

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

# Also dump machine-readable summary for the sidecar
print(f'# METRICS avg_ms={avg*1000:.3f} tflops={tflops:.2f} mem_bw_gbs={bw:.1f}')
" 2>&1)
echo "$GEMM_OUTPUT"
echo ""

# Parse metrics out for the JSON sidecar
GEMM_AVG_MS=$(echo "$GEMM_OUTPUT" | grep -oE "avg_ms=[0-9.]+" | cut -d= -f2 || echo null)
GEMM_TFLOPS=$(echo "$GEMM_OUTPUT" | grep -oE "tflops=[0-9.]+" | cut -d= -f2 || echo null)
MEM_BW_GBS=$(echo "$GEMM_OUTPUT" | grep -oE "mem_bw_gbs=[0-9.]+" | cut -d= -f2 || echo null)

# 9. Multi-GPU check
echo "=== GPU Count ==="
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $GPU_COUNT"
if [ "$GPU_COUNT" -gt 1 ]; then
    echo ""
    echo "=== All GPUs ==="
    nvidia-smi --query-gpu=index,name,clocks.gr,clocks.max.gr,memory.total,memory.free --format=csv,noheader
    echo ""
    echo "=== All GPU Identifiers (index, serial, UUID) ==="
    nvidia-smi --query-gpu=index,serial,uuid --format=csv,noheader
    echo ""
    echo "=== GPU-to-GPU Bandwidth (NVLink/PCIe) ==="
    nvidia-smi topo -m 2>/dev/null || echo "Topology info not available"
fi
echo ""

# 10. Location echo (already captured above, echo for readability)
echo "=== Pod Location ==="
echo "$POD_PUBLIC_IP — $POD_CITY, $POD_REGION, $POD_COUNTRY"
echo ""

# 11. GPU FINGERPRINT — prominent block so you can grep this across pods
echo "═══════════════════════════════════════════"
echo "  GPU FINGERPRINT (save this to find same pod again)"
echo "═══════════════════════════════════════════"
echo "pod_id=${POD_ID}"
echo "pod_dc_id=${POD_DC_ID}"
echo "pod_public_ip=${POD_PUBLIC_IP}"
echo "pod_location=${POD_CITY}, ${POD_REGION}, ${POD_COUNTRY}"
echo "gpu_count=${GPU_COUNT}"
echo "gemm_avg_ms=${GEMM_AVG_MS}"
echo "gemm_tflops=${GEMM_TFLOPS}"
echo "mem_bw_gbs=${MEM_BW_GBS}"
echo "--- per-gpu fingerprint ---"
nvidia-smi --query-gpu=index,name,serial,uuid,pci.bus_id --format=csv,noheader | \
    awk -F', ' '{printf "gpu[%s]: name=\"%s\" serial=%s uuid=%s pci=%s\n", $1, $2, $3, $4, $5}'
echo ""

# 12. JSON sidecar for machine-readable cross-run comparison
JSON_OUT="${JSON_OUT:-/tmp/pod-benchmark-$(date -u +%Y%m%d-%H%M%S).json}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Build per-GPU JSON via python to avoid quoting pain
python3 - "$JSON_OUT" "$TIMESTAMP" "$POD_ID" "$POD_HOSTNAME" "$POD_DC_ID" "$POD_PUBLIC_IP" \
        "$POD_CITY" "$POD_REGION" "$POD_COUNTRY" "$GEMM_AVG_MS" "$GEMM_TFLOPS" "$MEM_BW_GBS" \
        "$GPU_COUNT" <<'PYEOF'
import json, subprocess, sys

(out_path, ts, pod_id, hostname, dc, ip, city, region, country,
 gemm_ms, tflops, mem_bw, gpu_count) = sys.argv[1:14]

def _num(x):
    try: return float(x)
    except: return None

# Per-GPU identity
try:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,serial,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.max.gr",
         "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL)
    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "serial": parts[2],
                "uuid": parts[3],
                "pci_bus_id": parts[4],
                "pcie_gen_current": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None,
                "pcie_width_current": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None,
                "clocks_max_gr_mhz": _num(parts[7].replace(" MHz","")) if len(parts) > 7 else None,
            })
except Exception as e:
    gpus = [{"error": str(e)}]

# NVLink status summary
try:
    nvlink = subprocess.check_output(
        ["nvidia-smi", "nvlink", "--status"], text=True, stderr=subprocess.DEVNULL)
    nvlink_lines = [l.strip() for l in nvlink.strip().splitlines()]
except Exception:
    nvlink_lines = []

# CPU summary
try:
    lscpu = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    cpu = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip(); v = v.strip()
            if k in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)"):
                cpu[k] = v
except Exception:
    cpu = {}

result = {
    "schema_version": 1,
    "timestamp": ts,
    "runpod": {
        "pod_id": pod_id,
        "hostname": hostname,
        "dc_id": dc,
        "public_ip": ip,
    },
    "location": {"city": city, "region": region, "country": country},
    "gpu_count": int(gpu_count) if gpu_count.isdigit() else None,
    "gpus": gpus,
    "gemm": {
        "avg_ms": _num(gemm_ms),
        "tflops": _num(tflops),
        "shape": "4096x4096 bf16",
        "iterations": 20,
    },
    "memory_bandwidth_gbs": _num(mem_bw),
    "nvlink_status": nvlink_lines,
    "cpu": cpu,
}

with open(out_path, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
print(f"JSON sidecar written to: {out_path}")
PYEOF

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
echo ""
echo "  Sidecar JSON saved above. To find this exact GPU again later,"
echo "  grep for its uuid or serial across your saved benchmarks."
echo "═══════════════════════════════════════════"
