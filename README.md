# RunPod GPU Benchmark

A quick benchmark script that measures actual GPU performance on any RunPod pod before you commit to an expensive training run. Takes ~30 seconds.

Born out of the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition, where I discovered that GPU quality varies dramatically between pods, even pods with the same GPU model. A bad H100 can cost you 30% more in wasted compute time. This script tells you in 30 seconds whether to keep the pod or kill it.

---

> **Unofficial and personal.** This repository is my individual work on my own time and reflects my own opinion only. It is not affiliated with, endorsed by, or representative of RunPod, NVIDIA, OpenAI, or any employer or client.
>
> All benchmark data here comes from pods I rented personally between March and April 2026. Results will vary by pod, region, time of day, and the workload you are running. These numbers are indicative, not authoritative. Verify before using them to make budget decisions.
>
> Nothing in this repo is a recommendation to buy any specific product or service.

---

## Why This Exists

I was competing in OpenAI's Parameter Golf challenge (train the best 16MB language model in 10 minutes on 8xH100s). After running **13 RunPod pods across 5 regions and spending about $360**, I learned the hard way:

- **Not all H100s are equal.** GEMM performance varied from 0.19 ms (742 TFLOPS) to 0.70+ ms (about 195 TFLOPS) across different pods.
- **Location matters.** Iceland pods consistently outperformed generic US pods by about 16% (I still cannot explain why).
- **Clock throttling is silent.** Some pods report 1980 MHz max but run at 1400 MHz under load.
- **Bad pods waste money.** A 30-second benchmark saves hours of wasted compute at $2-22+/hr.

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
| **GEMM** | 4096x4096 bf16 matrix multiply (20 iterations) | **The single most important metric**: directly predicts training step time |
| **Memory Bandwidth** | 512MB element-wise operation | Predicts memory-bound operation speed |
| **GPU Count** | Number of GPUs + topology | Multi-GPU: verifies all GPUs present, shows NVLink topology |
| **Location** | Pod IP, city, region, country | Track which datacenters give best performance |

## How to Read the Results

### The GEMM Test (Most Important)

The GEMM (General Matrix Multiply) test runs a 4096x4096 bf16 matrix multiplication 20 times and reports the average time and TFLOPS.

**This is the single best predictor of training speed.** Deep learning training is dominated by matrix multiplications: attention layers, linear projections, MLP blocks. If your GEMM is slow, everything is slow.

```
GEMM 4096x4096 bf16: 0.19 ms  (742.2 TFLOPS)    ← Excellent
GEMM 4096x4096 bf16: 0.50 ms  (275.0 TFLOPS)    ← Acceptable
GEMM 4096x4096 bf16: 0.70 ms  (196.0 TFLOPS)    ← Bad, kill this pod
```

### Decision Framework

```
GEMM < 0.25 ms (> 550 TFLOPS)  →  KEEP - exceptional pod, run everything
GEMM 0.25-0.50 ms (275-550)    →  KEEP - good pod, fine for most workloads
GEMM 0.50-0.70 ms (196-275)    →  MAYBE - acceptable if no better pods available
GEMM > 0.70 ms (< 196 TFLOPS)  →  KILL - stop the pod and get a new one
```

### Clock Speed Check

```
Max GPU clock: 1980 MHz   ← Normal for H100 SXM
Max GPU clock: 1410 MHz   ← Throttled (overheating or power-limited)
```

If the current clock is significantly below max, the GPU is being throttled. This often happens with:
- Shared pods where other tenants are generating heat
- Pods in warm datacenters without adequate cooling
- Power-limited configurations

## Baseline Results

Real benchmark data collected across multiple RunPod pods, GPU types, and locations (March to April 2026) during personal pod rentals. Some data comes from the OpenAI Parameter Golf competition, the rest from independent training and inference experiments I ran on my own pods during the same period.

### Single GPU

| GPU | Location | GEMM (ms) | TFLOPS | Mem BW | Max Clock | PCIe | Cost/hr | Verdict |
|-----|----------|-----------|--------|--------|-----------|------|---------|---------|
| H100 80GB HBM3 | Kansas City, MO | **0.19** | **742** | 722 GB/s | 1980 MHz | Gen5x16 | $2.69 | EXCEPTIONAL |
| H100 80GB HBM3 | US (unknown dc) | ~0.55 | ~250 | — | — | — | ~$2.49 (community) | ACCEPTABLE |
| H100 80GB HBM3 | US (unknown dc) | ~0.55 | ~250 | — | — | — | — | ACCEPTABLE |
| H200 SXM 141GB HBM3e | Reykjavik, Iceland | — | — | — | — | — | ~$4.00 | Used for full BF16 Gemma 4 26B training (~$28 total, ~7h) |
| A40 48GB | Montreal, Canada | **4.66** (8192 bf16) | **108.8** | 557 GB/s | — | Gen4x16 | **$0.47** | EXCELLENT value; Gemma 2 27B Q5_K_M = 27.5 tok/s |
| RTX 5090 32GB | Reykjavik, Iceland | 0.76 | 180.6 | 426 GB/s | 3090 MHz | — | — | Blackwell desktop; **FA3 kernels fail on SM 12.0**, SDPA fallback works |

### Multi-GPU (8xH100 80GB SXM)

| Pod | Location | GEMM (ms) | TFLOPS | NVLink | Clock | CPU | Cost/hr | Verdict |
|-----|----------|-----------|--------|--------|-------|-----|---------|---------|
| IS-1 | Reykjavik, Iceland | **0.19** | **734** | 4x26.5 GB/s | 1980 MHz (all 8 GPUs at 96-99%) | Xeon 8468 (160 threads) | ~$21.52 | EXCEPTIONAL |
| KC-2 | Kansas City, MO | 0.19 | ~740 | 4x26.5 GB/s | 1980 MHz | Xeon 8470 | ~$21.52 | EXCELLENT (second set of 8xH100 runs) |

### Per-GPU TFLOPS varies 4x across pods

```
Iceland 8xH100 (IS-1):  734 TFLOPS per GPU  (best)
KC 1xH100 (Pod 3):      742 TFLOPS per GPU  (matches IS-1)
Generic US 1xH100:      ~250 TFLOPS per GPU (throttled or old hardware)
RTX 5090 (Iceland):     180 TFLOPS          (consumer tier, 32GB VRAM)
A40 (Montreal):         108 TFLOPS          (older architecture but great $/TFLOPS)
```

### Training Step Times by Pod Quality

Same code, same model architecture, different pods. This is why GEMM is the metric that matters:

| Pod | GPU Quality | Code | Step Time | Steps in 10 min | Relative Speed |
|-----|------------|------|-----------|-----------------|----------------|
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #77 LoRA TTT | 51 ms/step | ~11,800 | **1.0x (fastest)** |
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #462 | 69-72 ms/step | ~8,300 | 0.74x |
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #505 | 133 ms/step | ~4,500 | 0.38x |
| KC-1 (KC 1x) | 742 TFLOPS | PR #406 | 568-572 ms/step | ~525 | 0.044x |
| US-1 (US 1x) | ~250 TFLOPS | PR #406 | ~580 ms/step | ~515 | 0.043x |
| RTX 5090 (Iceland 1x) | 180 TFLOPS | PR #414 | 326 ms/step | ~1,800 | 0.16x |
| RTX 5090 (Iceland 1x) | 180 TFLOPS | PR #549 | 385 ms/step | ~1,550 | 0.13x |

**Key insight:** An 8xH100 pod is not just 8x faster: the DDP parallelism and NVLink interconnect make it 10-20x faster per step. The 1xH100 pod gets about 525 steps in 5 minutes while the 8xH100 pod gets about 11,800 steps in 10 minutes.

### Competition Results by Pod

Actual competition scores (BPB = bits per byte, lower is better) achieved on different pods. These are from OpenAI Parameter Golf runs where the same pipeline was re-executed across different hardware:

| Pod | Config | val_bpb | Step Time | Model Size | Under 16MB? |
|-----|--------|---------|-----------|------------|-------------|
| IS-1 8xH100 | PR #462 KV=8 MLP=1792 + 10ep TTT | **1.0689** | 72 ms | 19.2 MB | No |
| IS-1 8xH100 | PR #462 KV=4 MLP=1536 DIM=496 | **1.0935** | 70 ms | 15.37 MB | **Yes** (submission candidate) |
| IS-1 8xH100 | PR #505 no TTT | 1.1279 | 133 ms | 19.8 MB | No |
| IS-1 8xH100 | PR #77 LoRA TTT | 1.1951 | 51 ms | 15.9 MB | Yes |
| KC-1 1xH100 | PR #406 baseline | 2.1809 | 572 ms | ~15 MB | Yes |
| KC-1 1xH100 | PR #406 + reduce-overhead | 2.1787 | 568 ms | ~15 MB | Yes |
| RTX 5090 (Iceland 1x) | PR #414 100 steps | 2.3155 @ step 100 | 326 ms | N/A (100-step diagnostic) | Yes |

## Non-competition pod data

Parallel to the Parameter Golf work, I used the same benchmark on other training and inference workloads I ran on my own pods. Including these for breadth since the cost-per-value profile is very different from record-chasing training:

| Workload | GPU | Location | Measured | Cost | Notes |
|----------|-----|----------|----------|------|-------|
| Full BF16 fine-tune of a ~27B parameter open model | H200 SXM | Reykjavik, Iceland | 50-65% GPU util over 6.9 h | ~$28 total | Full precision fit in 141 GB without quantization |
| Quantized inference eval on a ~27B open model | A40 48GB | Montreal, Canada | 27.5 tok/s, GEMM 108.8 TFLOPS | ~$0.47/hr | Community cloud; good $/tok for quantized inference |
| Quantized inference eval on a ~32B open model | A40 48GB | Montreal, Canada | queued after the 27B run | ~$0.47/hr | Same pod, queued sequentially |

**Key observation:** For quantized inference of 27-70B class open models, A40 community cloud came out dramatically cheaper than H100 or H200 in my runs, and the speed difference was much smaller than the cost difference. For full-precision fine-tuning of the same model size, the VRAM difference (46 GB vs 141 GB) mattered more to me than raw TFLOPS did.

## Blackwell GPUs (RTX 5090, DGX Spark)

Blackwell (SM 12.x) pods and dev boxes need special handling:

- **Flash Attention 3** (`flash_attn_interface`) builds on Blackwell but the compiled kernels target only SM 80 and SM 90. At runtime on SM 12.x you get `no kernel image is available for execution on the device` even after a successful `pip install -e .`. This has been reported to NVIDIA ([developer forum thread](https://forums.developer.nvidia.com/t/i-keep-failing-to-install-flash-attention-3-in-the-ltx-2-uv-environment/357560)) and upstream ([Dao-AILab/flash-attention#1969](https://github.com/Dao-AILab/flash-attention/issues/1969)).
- **Flash Attention 2** (`flash_attn` 2.7.4) is pre-installed in the NGC PyTorch container (`nvcr.io/nvidia/pytorch:26.03-py3`) and runs bit-exact to `torch.nn.functional.scaled_dot_product_attention` on Blackwell. Use FA2 or SDPA, not FA3.
- **Triton shared memory** on Blackwell (about 101 KB per SM on GB10) is less than half of Hopper (about 228 KB). Many modded-nanogpt style kernels hit `OutOfResources: shared memory` at their default block sizes. Reduce `BLOCK_SIZE_N` and `num_stages` or the kernel will not launch.
- **CUDA 13** and sm_121 full native support is rolling out through 2026. See [PyTorch forums](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744) and [vLLM #31128](https://github.com/vllm-project/vllm/issues/31128) for current state.

## Cost summary (from real pod spend)

13 pods across 5 regions, about $360 total spend on the Parameter Golf competition alone. Broken down:

| Segment | Spend | Notes |
|---------|-------|-------|
| Iceland 8xH100 experiments | ~$65 | 3 hours, 10 experiments |
| KC 1xH100 baseline testing | ~$12 | PR #406 plus T1-T5 |
| KC 8xH100 secondary runs | ~$55 | Re-verification plus 3-seed competition runs |
| March 23-26 early pods (US) | ~$40 | Mostly throttled H100s; taught me to benchmark first |
| RTX 5090 testing (Iceland) | ~$6 | Diagnostic only, 1.5 hours |
| Compliance H200 training (Iceland) | ~$28 | Full Gemma 4 26B BF16 training |
| Compliance A40 inference (Montreal) | ~$1.20 | 2.5 hours on quantized eval |
| Misc overnight pods + failures | ~$150 | Includes the really bad throttled pods I wish I had killed earlier |

A 30-second benchmark could have saved a meaningful fraction of the last row.

## Tips for Getting Good Pods

1. **Try Iceland.** In my testing, Icelandic datacenter pods (EUR-IS region) consistently delivered 0.19 ms GEMM on H100 SXM. The phenomenon repeats across H100 1x, H100 8x, and H200 SXM. It does not appear on RTX 5090 (Blackwell is already at its stock clock, not throttled).

2. **Run the benchmark immediately.** Do not start a 2-hour training run before checking. The 30-second benchmark has saved me about $40 multiple times.

3. **Check all GPUs on multi-GPU pods.** Sometimes 7 of 8 GPUs are fine and one is throttled. The script checks all GPUs when it detects more than one.

4. **Kill bad pods fast.** RunPod charges by the minute. A bad pod at $21.52/hr costs $0.36/min. If the benchmark looks bad, kill it immediately and spin up a new one.

5. **For 8xH100 training work, request Iceland explicitly if possible.** The Iceland pod NVLink topology (4 links at 26.5 GB/s per GPU, all 8 GPUs at 1980 MHz sustained) is the combination that made the 10-20x per-step speedup possible in our runs.

6. **For inference of quantized 27-70B models, A40 community cloud is the sweet spot.** $0.47/hr vs $4/hr on H200 and the speed difference is much smaller than 8.5x.

7. **Save your benchmark results.** Redirect output to a file so you can compare across pods:
   ```bash
   bash pod-test.sh | tee benchmark-$(date +%Y%m%d-%H%M%S).txt
   ```

8. **Network volumes persist.** If you are running repeated experiments, use a RunPod network volume to persist your data and checkpoints. Container storage is lost when the pod is killed.

9. **If your code uses FA3, check the target GPU first.** FA3 is Hopper-only in the shipping binaries. Blackwell (RTX 5090, GB10 DGX Spark, B200 where FA3 support is still rolling out) needs FA2 or SDPA. See the Blackwell section above.

## Finding a specific GPU again

If a pod turns out to be exceptional, the script now captures everything you need to request the same hardware later. It writes two things:

1. A **GPU FINGERPRINT** block to stdout with pod id, datacenter, per-GPU UUID and serial, and the actual GEMM and memory bandwidth numbers.
2. A **JSON sidecar** (default `/tmp/pod-benchmark-<timestamp>.json`, override with `JSON_OUT=path.json`) containing the same data in a machine-readable form.

Save these alongside your training logs so you can cross-reference later.

### What's captured

```
pod_id=<runpod pod id, e.g. a8f3c2b1e7>
pod_dc_id=<datacenter, e.g. EUR-IS-1>
pod_public_ip=<ip:port>
pod_location=<city, region, country>
gemm_tflops=<measured>
mem_bw_gbs=<measured>
gpu[0]: name="NVIDIA H100 80GB HBM3" serial=<13-digit> uuid=GPU-<hex uuid> pci=<bus id>
gpu[1]: ...
...
```

UUID is unique to the physical GPU. Two rentals with the same UUID mean you got the same card back.

### The realistic workflow

RunPod does not let you request a specific GPU by UUID directly. What you can do:

1. **From the fingerprint**, note the `pod_dc_id` and the GPU model that gave you the performance you want. These are the levers you have.
2. **Rent a pod in that specific datacenter** with the same GPU type. Use the RunPod CLI or API with `--dataCenterId` pinning:
   ```bash
   runpodctl pod create \
     --name "chasing-good-gpu" \
     --gpu "NVIDIA H100 80GB HBM3" \
     --gpuCount 8 \
     --dataCenterId "EUR-IS-1" \
     --templateId "<your PyTorch template>"
   ```
3. **Run this benchmark** on the new pod. Compare the fingerprint JSON to your saved one.
4. **If the UUID matches**, keep the pod. That's the same physical card.
5. **If the UUID does not match but performance is equivalent**, this is a peer card in the same DC. Often good enough.
6. **If performance is degraded**, stop the pod and retry. RunPod charges by the minute, so this loop costs about $0.10 per attempt on a single H100 and $0.35 per attempt on an 8xH100.

### Listing available datacenters

```bash
runpodctl dc list                    # human table
runpodctl dc list -o json            # machine readable
runpodctl gpu list --include-unavailable   # what GPU types exist where
```

The `dc_id` field from the sidecar maps directly to the `--dataCenterId` flag.

### Building a personal "known good pods" registry

Once you have a handful of benchmarks saved:

```bash
# Find all your sidecar files
ls -1 /tmp/pod-benchmark-*.json ~/bench-archive/*.json

# Pull out every UUID and its measured TFLOPS
jq -r '[.timestamp, .runpod.dc_id, .gpus[0].uuid, .gemm.tflops] | @tsv' \
    /tmp/pod-benchmark-*.json 2>/dev/null | sort -k4 -nr

# Or grep if jq isn't installed
grep -H "uuid\|tflops" /tmp/pod-benchmark-*.json
```

Over enough rentals, you will start to see specific UUIDs reappear — those are your known-good targets, and the `dc_id` field tells you where to rent to get them back.

## Sample Output

```
═══════════════════════════════════════════
  RunPod GPU Benchmark
═══════════════════════════════════════════

=== GPU Identity ===
NVIDIA H100 80GB HBM3, 00000000:1B:00.0, 1234567890123, GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

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
  GPU FINGERPRINT (save this to find same pod again)
═══════════════════════════════════════════
pod_id=a8f3c2b1e7
pod_dc_id=EUR-IS-1
pod_public_ip=213.181.122.175:10750
pod_location=Reykjavik, Capital Region, IS
gpu_count=1
gemm_avg_ms=0.19
gemm_tflops=742.2
mem_bw_gbs=722
--- per-gpu fingerprint ---
gpu[0]: name="NVIDIA H100 80GB HBM3" serial=1234567890123 uuid=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx pci=00000000:1B:00.0

JSON sidecar written to: /tmp/pod-benchmark-20260421-201534.json

═══════════════════════════════════════════
  BENCHMARK COMPLETE

  REFERENCE VALUES (good H100 SXM):
  GEMM: < 0.50 ms (> 275 TFLOPS)
  Memory BW: > 2800 GB/s
  Max GPU clock: 1980 MHz

  If GEMM > 0.70 ms or BW < 2000 GB/s,
  consider stopping and getting a new pod.

  Sidecar JSON saved above. To find this exact GPU again later,
  grep for its uuid or serial across your saved benchmarks.
═══════════════════════════════════════════
```

## Contributing

If you have run this benchmark on different GPU types (A100, A6000, L40S, H200, B200, etc.) or in different RunPod regions, I would love to add your data to the baseline table. Open a PR or issue with your benchmark output.

### Particularly wanted

- **B200** runtime benchmark (RunPod lists it but I have not rented one yet)
- **L40S** results, both for training and inference
- **A6000** multi-GPU NVLink topology
- **Additional region data points** outside US and Iceland (APAC, EU-DE, CA-MTL)
- **AMD Instinct MI300X / MI355X** results (separate fork welcome; this script uses `nvidia-smi` only)

## License

MIT

## Author

**Nathan Maine** ([@NathanMaine on GitHub](https://github.com/NathanMaine))

This repository is a personal project on my own time. It is not affiliated with and does not represent the position of any employer, client, or organization.

Built during the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition. [Issue #396](https://github.com/karpathy/autoresearch/issues/396) on karpathy/autoresearch was filed during this same adventure.

---

### Related

- **[Parameter Golf Experiment Lab](https://github.com/NathanMaine/parameter-golf-experiment-lab)** — Interactive dashboard visualizing 352 submissions and 46+ experiments from the competition
- **[NVIDIA Developer Forum: Flash Attention 3 on DGX Spark](https://forums.developer.nvidia.com/t/i-keep-failing-to-install-flash-attention-3-in-the-ltx-2-uv-environment/357560)** — Thread where I documented the FA3 build success / runtime failure pattern on Blackwell SM 12.1
