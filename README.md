# RunPod GPU Benchmark

A quick benchmark script that measures actual GPU performance on any RunPod pod before you commit to an expensive training run. Takes ~30 seconds.

Born out of the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition, where I discovered that GPU quality varies dramatically between pods — even pods with the same GPU model. A "bad" H100 can cost you 30% more in wasted compute time. This script tells you in 30 seconds whether to keep the pod or kill it.

## Why This Exists

I was competing in OpenAI's Parameter Golf challenge (train the best 16MB language model in 10 minutes on 8×H100s). After spending $256 across multiple pods, I learned the hard way:

- **Not all H100s are equal.** GEMM performance varied from 0.19ms (740 TFLOPS) to 0.70ms+ (~195 TFLOPS) across different pods.
- **Location matters.** Iceland pods consistently outperformed US pods by ~16% (I still can't explain why).
- **Clock throttling is silent.** Some pods report 1980 MHz max but run at 1400 MHz under load.
- **Bad pods waste money.** A 30-second benchmark saves hours of wasted compute at $2-20+/hr.

## Quick Start

SSH into your RunPod pod, then:

```bash
# Option 1: Download and run directly
curl -sL https://raw.githubusercontent.com/NathanMaine/runpod-gpu-benchmark/main/pod-test.sh | bash

# Option 2: Clone and run
git clone https://github.com/NathanMaine/runpod-gpu-benchmark.git
cd runpod-gpu-benchmark
bash pod-test.sh
```

No dependencies beyond what's already on a RunPod PyTorch template (`nvidia-smi`, `python3`, `torch`, `curl`, `dd`).

## What It Measures

| Test | What It Tells You | Why It Matters |
|------|-------------------|----------------|
| **GPU Identity** | GPU model, serial, UUID | Verify you got what you're paying for |
| **Clock Speeds** | Current vs max GPU/memory clocks | Detect thermal throttling |
| **Memory & PCIe** | VRAM total/free, PCIe generation/width | Ensure full VRAM available, no bottlenecks |
| **NVLink** | Inter-GPU link status and bandwidth | Critical for multi-GPU DDP training |
| **CPU** | Model, core count, thread count | Affects data loading pipeline |
| **Disk Speed** | Sequential write bandwidth (dd) | Slow disk = slow checkpoint saves |
| **Network** | Download speed to HuggingFace | Affects model/dataset download time |
| **GEMM** | 4096×4096 bf16 matrix multiply (20 iterations) | **The single most important metric** — directly predicts training step time |
| **Memory Bandwidth** | 512MB element-wise operation | Predicts memory-bound operation speed |
| **GPU Count** | Number of GPUs + topology | Multi-GPU: verifies all GPUs present, shows NVLink topology |
| **Location** | Pod IP, city, region, country | Track which datacenters give best performance |

## How to Read the Results

### The GEMM Test (Most Important)

The GEMM (General Matrix Multiply) test runs a 4096×4096 bf16 matrix multiplication 20 times and reports the average time and TFLOPS.

**This is the single best predictor of training speed.** Deep learning training is dominated by matrix multiplications — attention layers, linear projections, MLP blocks. If your GEMM is slow, everything is slow.

```
GEMM 4096x4096 bf16: 0.19 ms  (742.2 TFLOPS)    ← Excellent
GEMM 4096x4096 bf16: 0.50 ms  (275.0 TFLOPS)    ← Acceptable
GEMM 4096x4096 bf16: 0.70 ms  (196.0 TFLOPS)    ← Bad — kill this pod
```

### Decision Framework

```
GEMM < 0.25 ms (> 550 TFLOPS)  →  KEEP — exceptional pod, run everything
GEMM 0.25-0.50 ms (275-550)    →  KEEP — good pod, fine for most workloads
GEMM 0.50-0.70 ms (196-275)    →  MAYBE — acceptable if no better pods available
GEMM > 0.70 ms (< 196 TFLOPS)  →  KILL — stop the pod and get a new one
```

### Clock Speed Check

```
Max GPU clock: 1980 MHz   ← Normal for H100 SXM
Max GPU clock: 1410 MHz   ← Throttled — overheating or power-limited
```

If the current clock is significantly below max, the GPU is being throttled. This often happens with:
- Shared pods where other tenants are generating heat
- Pods in warm datacenters without adequate cooling
- Power-limited configurations

## Baseline Results

Real benchmark data collected during the OpenAI Parameter Golf competition (March 2026) across multiple RunPod pods and locations.

### Single GPU (1×H100 80GB SXM)

| Pod | Location | GEMM (ms) | TFLOPS | Mem BW | Max Clock | PCIe | Verdict |
|-----|----------|-----------|--------|--------|-----------|------|---------|
| KC-1 | Kansas City, MO | **0.19** | **742** | 722 GB/s | 1980 MHz | Gen5×16 | EXCEPTIONAL |
| US-1 | US (unknown) | ~0.55 | ~250 | — | — | — | ACCEPTABLE |
| US-2 | US (unknown) | ~0.55 | ~250 | — | — | — | ACCEPTABLE |

### Multi-GPU (8×H100 80GB SXM)

| Pod | Location | GEMM (ms) | TFLOPS | NVLink | Clock | Cost/hr | Verdict |
|-----|----------|-----------|--------|--------|-------|---------|---------|
| IS-1 | Reykjavík, Iceland | **0.19** | **734** | 4×26.5 GB/s | 1980 MHz | ~$21.52 | EXCEPTIONAL |

### Training Step Times by Pod Quality

This table shows how pod quality directly affects training speed. Same code, same model architecture, different pods:

| Pod | GPU Quality | Code | Step Time | Steps in 10 min | Relative Speed |
|-----|------------|------|-----------|-----------------|----------------|
| IS-1 (Iceland 8×) | 734 TFLOPS | PR #77 | 51 ms/step | ~11,800 | **1.0× (fastest)** |
| IS-1 (Iceland 8×) | 734 TFLOPS | PR #462 | 69 ms/step | ~8,700 | 0.74× |
| IS-1 (Iceland 8×) | 734 TFLOPS | PR #505 | 133 ms/step | ~4,500 | 0.38× |
| KC-1 (KC 1×) | 742 TFLOPS | PR #406 | 572 ms/step | ~525 | 0.044× |
| US-1 (US 1×) | ~250 TFLOPS | PR #406 | ~580 ms/step | ~515 | 0.043× |

**Key insight:** An 8×H100 pod is not just 8× faster — the DDP parallelism and NVLink interconnect make it 10-20× faster per step. The 1× pod gets ~525 steps in 5 minutes while the 8× pod gets ~11,800 steps in 10 minutes.

### Competition Results by Pod

These are actual competition scores (BPB = bits per byte, lower is better) achieved on different pods:

| Pod | Config | val_bpb | Step Time | Model Size | Under 16MB? |
|-----|--------|---------|-----------|------------|-------------|
| IS-1 8×H100 | PR #462 KV=8 MLP=1792 + 10ep TTT | **1.0689** | 72ms | 19.2MB | No |
| IS-1 8×H100 | PR #462 KV=4 MLP=1536 DIM=496 | **1.0935** | 70ms | 15.37MB | **Yes** |
| IS-1 8×H100 | PR #505 no TTT | 1.1279 | 133ms | 19.8MB | No |
| IS-1 8×H100 | PR #77 LoRA TTT | 1.1951 | 51ms | 15.9MB | Yes |
| KC-1 1×H100 | PR #406 baseline | 2.1809 | 572ms | ~15MB | Yes |
| KC-1 1×H100 | PR #406 + reduce-overhead | 2.1787 | 568ms | ~15MB | Yes |

## Tips for Getting Good Pods

1. **Try Iceland.** In my testing, Icelandic datacenter pods (EUR-IS region) consistently delivered 0.19ms GEMM. I cannot explain why, but the data is clear.

2. **Run the benchmark immediately.** Don't start a 2-hour training run before checking. The 30-second benchmark can save you $40+.

3. **Check all GPUs on multi-GPU pods.** Sometimes 7 of 8 GPUs are fine and one is throttled. The script checks all GPUs when it detects more than one.

4. **Kill bad pods fast.** RunPod charges by the minute. A bad pod at $21.52/hr costs $0.36/min. If the benchmark looks bad, kill it immediately and spin up a new one.

5. **Save your benchmark results.** Redirect output to a file so you can compare across pods:
   ```bash
   bash pod-test.sh | tee benchmark-$(date +%Y%m%d-%H%M%S).txt
   ```

6. **Network volumes persist.** If you're running repeated experiments, use a RunPod network volume to persist your data/checkpoints. Container storage is lost when the pod is killed.

## Sample Output

```
═══════════════════════════════════════════
  RunPod GPU Benchmark
═══════════════════════════════════════════

=== GPU Identity ===
NVIDIA H100 80GB HBM3, 00000000:1B:00.0, 1650724018113, GPU-74fd4d05-2bcc-aa9e-2691-5c3f13260073

=== Clock Speeds ===
1980 MHz, 1980 MHz, 2619 MHz, 2619 MHz
(graphics_clock, max_graphics, mem_clock, max_mem) in MHz

=== Memory & PCIe ===
81559 MiB, 81071 MiB, 5, 16

=== NVLink ===
GPU 0: NVLink link 0: <active>
GPU 0: NVLink link 1: <active>
GPU 0: NVLink link 2: <active>
GPU 0: NVLink link 3: <active>

=== CPU ===
Model name:            Intel(R) Xeon(R) Platinum 8470
CPU(s):                208
Thread(s) per core:    2

=== Disk Speed ===
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.289394 s, 3.7 GB/s

=== Network ===
Download speed: 2485760.000 bytes/sec

=== GPU Compute (GEMM) ===
GEMM 4096x4096 bf16: 0.19 ms  (742.2 TFLOPS)
Min: 0.18 ms  Max: 0.20 ms
Memory bandwidth: 722 GB/s

=== GPU Count ===
GPUs detected: 1

=== Pod Location ===
64.247.201.52 — Kansas City, Missouri, US

═══════════════════════════════════════════
  BENCHMARK COMPLETE

  REFERENCE VALUES (good H100 SXM):
  GEMM: < 0.50 ms (> 275 TFLOPS)
  Memory BW: > 2800 GB/s
  Max GPU clock: 1980 MHz

  If GEMM > 0.70 ms or BW < 2000 GB/s,
  consider stopping and getting a new pod.
═══════════════════════════════════════════
```

## Contributing

If you've run this benchmark on different GPU types (A100, A6000, L40S, H200, etc.) or in different RunPod regions, I'd love to add your data to the baseline table. Open a PR or issue with your benchmark output.

## License

MIT

## Author

**Nathan Maine** — [Memoriant Inc.](https://memoriant.ai)
NVIDIA Inception Member | Building AI compliance tools for defense contractors

Built during the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition. [Issue #396](https://github.com/karpathy/autoresearch/issues/396) on karpathy/autoresearch was filed during this same adventure.

---

### Related

- **[Parameter Golf Experiment Lab](https://github.com/NathanMaine/parameter-golf-experiment-lab)** — Interactive dashboard visualizing 352 submissions and 46+ experiments from the competition
