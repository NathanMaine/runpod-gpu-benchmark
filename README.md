# RunPod GPU Benchmark

A small benchmark script I wrote to measure what I was actually getting on any RunPod pod before I committed to an expensive training run. Takes around 30 seconds on my machine. This repo is a personal record of what I ran and what I saw.

I started writing it during the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition, when I noticed (in my own rentals) that GPU performance seemed to vary a lot between pods, even pods listed with the same GPU model. The script is my attempt at a quick sanity check.

---

> **Unofficial and personal.** This repository is my individual work on my own time and reflects my own opinion only. It is not affiliated with, endorsed by, or representative of RunPod, NVIDIA, OpenAI, or any employer or client.
>
> **Nothing in this document is stated as fact or as an authoritative claim.** Every number, table, and observation below is a personal measurement from a pod I personally rented between March and April 2026, and every interpretation is my own read of that small sample. I am not an authority on GPU performance, cloud hardware, or RunPod inventory. Your pod, your day, your workload, and your tooling will differ.
>
> All benchmark data here comes from pods I rented personally. Results will vary. Treat these numbers as one person's observations, not guidance. Verify before using any of it to make budget decisions.
>
> Nothing in this repo is a recommendation to buy any specific product or service.

---

## Why I started keeping this record

I was competing in OpenAI's Parameter Golf challenge (train the best 16MB language model in 10 minutes on 8xH100s). I rented roughly 13 RunPod pods across 5 regions on my own time and spent something in the neighborhood of $360 of my own money. Over those rentals I noticed a few things in my own runs:

- The H100s I rented didn't all behave the same in my GEMM measurements. Among my pods, one measurement came back at 0.19 ms (about 742 TFLOPS by my math), and others came back in the 0.70+ ms range (roughly 195 TFLOPS). I don't know what caused the spread — I'm just reporting what my script printed.
- In my tiny sample, Iceland pods tended to post faster GEMM numbers than a couple of generic US ones I happened to land on. I can't explain why and I can't generalize from the small number I ran.
- On some pods the reported max clock was 1980 MHz but what I actually observed under load was lower. I took that as a sign the card might be throttling, but I have not independently verified cause.
- When a pod measured slowly in my benchmark and I kept it anyway, I spent more per training step. That is where the impulse to measure first came from.

This script is what I now run on every new pod. Your experience and conclusions may differ.

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

The RunPod PyTorch templates I've used all had `nvidia-smi`, `python3`, `torch`, `curl`, and `dd` available. If yours is different, the script will likely need adjustment.

## What the script tries to measure

These are the signals I personally pay attention to. None of them are the full picture on their own, and I may be weighting them wrong.

| Test | What I read from it | Why I look at it |
|------|---------------------|------------------|
| GPU Identity | Name, serial, UUID, PCI bus id | Confirms the card I was allocated; lets me compare across rentals |
| Clock Speeds | Current vs max GPU/memory clocks | Big gap between current and max has, in my rentals, correlated with slow runs |
| Memory and PCIe | VRAM total/free, PCIe generation/width | I want to know what I actually got vs what was listed |
| NVLink | Inter-GPU link status | For the multi-GPU pods I rented, links that came back inactive predicted slow DDP in my runs |
| CPU | Model, core count, thread count | In my runs, data loading speed tracked with these |
| Disk Speed | Sequential write via `dd` | Checkpoint save time on my runs seemed to scale with this |
| Network | Download from HuggingFace | How long I waited on first-time model and dataset downloads |
| GEMM | 4096x4096 bf16 matmul, 20 iters | The one I personally pay most attention to (more below) |
| Memory Bandwidth | 512MB element-wise op | My rough proxy for memory-bound kernel speed |
| GPU Count | Number of GPUs and topology | Sanity-check that I got what I paid for |
| Location | Pod IP, city, region, country | I keep this in my notes so I can correlate pod-by-location over time |

## How I read the numbers

### Why I pay most attention to the GEMM number

The GEMM test runs a 4096x4096 bf16 matrix multiply 20 times and averages the time.

In my own training runs, my step times roughly tracked with this GEMM number. Attention layers, linear projections, and MLP blocks are dominated by matmuls on the kind of models I was training, so this matched my intuition. I am not claiming it's the "right" metric for your workload — it's the one I've been using.

What I've personally seen in my runs:

```
GEMM 4096x4096 bf16: 0.19 ms  (742.2 TFLOPS)    ← what I'd keep and run on
GEMM 4096x4096 bf16: 0.50 ms  (275.0 TFLOPS)    ← still fine for my workloads
GEMM 4096x4096 bf16: 0.70 ms  (196.0 TFLOPS)    ← I've personally killed pods at this level
```

### How I personally decide whether to keep a pod

My own rough decision rule, not a rule anyone else should follow:

```
GEMM < 0.25 ms (> 550 TFLOPS)  →  I keep it
GEMM 0.25-0.50 ms (275-550)    →  I keep it
GEMM 0.50-0.70 ms (196-275)    →  I keep if nothing better is available
GEMM > 0.70 ms (< 196 TFLOPS)  →  I typically stop the pod and retry
```

This is tuned for the specific workload I was running (parameter-golf style training). Your workload may care about totally different things.

### How I think about clock speed

```
Max GPU clock: 1980 MHz   ← what I saw on the H100 SXMs I rented that felt fast
Max GPU clock: 1410 MHz   ← what I saw on pods that felt slow; I guessed throttling
```

I don't have a way to verify whether the gap is actually thermal or power-related from inside the pod. It's just a signal I've been using.

## What I measured on my own rentals

These are the numbers my script printed on pods I rented between March and April 2026, sometimes during Parameter Golf work, sometimes during personal training and inference experiments. They are indicative of one person's experience with a small number of pods and one point in time. I am not claiming they generalize.

### Single GPU (what I saw on my own pods)

| GPU | Location | GEMM (ms) | TFLOPS | Mem BW | Max Clock | PCIe | Cost/hr | My personal read |
|-----|----------|-----------|--------|--------|-----------|------|---------|------------------|
| H100 80GB HBM3 | Kansas City, MO | 0.19 | 742 | 722 GB/s | 1980 MHz | Gen5x16 | $2.69 | Felt exceptional in my runs |
| H100 80GB HBM3 | US (unknown dc) | ~0.55 | ~250 | — | — | — | ~$2.49 (community) | Worked fine for me |
| H100 80GB HBM3 | US (unknown dc) | ~0.55 | ~250 | — | — | — | — | Similar to the other US one |
| H200 SXM 141GB HBM3e | Reykjavik, Iceland | — | — | — | — | — | ~$4.00 | What I used for full-BF16 fine-tuning of a ~27B model; the run I ran cost me ~$28 total over ~7 hours |
| A40 48GB | Montreal, Canada | 4.66 (8192 bf16) | 108.8 | 557 GB/s | — | Gen4x16 | $0.47 | My personal sweet spot for quantized inference on open 27-70B models; Gemma 2 27B Q5_K_M ran at 27.5 tok/s in my test |
| RTX 5090 32GB | Reykjavik, Iceland | 0.76 | 180.6 | 426 GB/s | 3090 MHz | — | — | Blackwell desktop. In my runs I could not get FA3 kernels to execute on SM 12.0; SDPA fallback worked. Treat this as a single data point from one rental. |

### Multi-GPU (8xH100 80GB SXM, pods I rented)

| Pod | Location | GEMM (ms) | TFLOPS | NVLink | Clock | CPU | Cost/hr | My personal read |
|-----|----------|-----------|--------|--------|-------|-----|---------|------------------|
| IS-1 | Reykjavik, Iceland | 0.19 | 734 | 4x26.5 GB/s | 1980 MHz on all 8 GPUs under load | Xeon 8468 (160 threads) | ~$21.52 | The fastest pod I personally rented |
| KC-2 | Kansas City, MO | 0.19 | ~740 | 4x26.5 GB/s | 1980 MHz | Xeon 8470 | ~$21.52 | Indistinguishable from IS-1 in my runs |

### What the per-GPU TFLOPS variance looked like in my data

The spread across a handful of my own rentals was wider than I expected:

```
Iceland 8xH100 (IS-1):  734 TFLOPS per GPU  (my fastest)
KC 1xH100 (Pod 3):      742 TFLOPS per GPU  (matched IS-1 in my run)
Generic US 1xH100:      ~250 TFLOPS per GPU (noticeably slower in my runs)
RTX 5090 (Iceland):     180 TFLOPS          (consumer card, 32GB VRAM)
A40 (Montreal):         108 TFLOPS          (older gen; in my opinion great dollar-per-TFLOPS for what I used it for)
```

I don't know why the US 1xH100 pods I happened to rent were slower. It could be throttling, older hardware revs, shared hosts, my bad luck, or something I'm not seeing. I'm not drawing a general conclusion — just recording what my script printed.

### Training step times in my own runs

Same code, same model, different pods. This is where the GEMM spread showed up for me at the step level. Just my runs:

| Pod | GPU (per-GPU TFLOPS I measured) | Code | Step Time I saw | Steps in 10 min | My relative speed |
|-----|--------------------------------|------|-----------------|-----------------|--------------------|
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #77 LoRA TTT | 51 ms/step | ~11,800 | fastest in my set |
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #462 | 69-72 ms/step | ~8,300 | 0.74x of above |
| IS-1 (Iceland 8x) | 734 TFLOPS | PR #505 | 133 ms/step | ~4,500 | 0.38x |
| KC-1 (KC 1x) | 742 TFLOPS | PR #406 | 568-572 ms/step | ~525 | 0.044x |
| US-1 (US 1x) | ~250 TFLOPS | PR #406 | ~580 ms/step | ~515 | 0.043x |
| RTX 5090 (Iceland 1x) | 180 TFLOPS | PR #414 | 326 ms/step | ~1,800 | 0.16x |
| RTX 5090 (Iceland 1x) | 180 TFLOPS | PR #549 | 385 ms/step | ~1,550 | 0.13x |

My personal interpretation (could be wrong): the 1xH100 to 8xH100 jump was much more than 8x for me. DDP overhead on a single GPU plus not using NVLink seemed to leave a lot on the table. Same code, different pod count, roughly 20x difference in steps per minute in my runs.

### Competition scores I personally achieved on each pod

BPB = bits per byte, lower is better. These are my own runs of public code on pods I personally rented:

| Pod | Config | My val_bpb | My step time | Model size | Under 16MB? |
|-----|--------|------------|--------------|------------|-------------|
| IS-1 8xH100 | PR #462 KV=8 MLP=1792 + 10ep TTT | 1.0689 | 72 ms | 19.2 MB | No |
| IS-1 8xH100 | PR #462 KV=4 MLP=1536 DIM=496 | 1.0935 | 70 ms | 15.37 MB | Yes (I used this as a submission candidate) |
| IS-1 8xH100 | PR #505 no TTT | 1.1279 | 133 ms | 19.8 MB | No |
| IS-1 8xH100 | PR #77 LoRA TTT | 1.1951 | 51 ms | 15.9 MB | Yes |
| KC-1 1xH100 | PR #406 baseline | 2.1809 | 572 ms | ~15 MB | Yes |
| KC-1 1xH100 | PR #406 + reduce-overhead | 2.1787 | 568 ms | ~15 MB | Yes |
| RTX 5090 (Iceland 1x) | PR #414 100 steps | 2.3155 @ step 100 | 326 ms | N/A (diagnostic) | Yes |

## Other workloads I ran on my own pods

Parallel to the Parameter Golf runs, I used the same benchmark on a couple of training and inference experiments of my own. Reporting for breadth because the cost-per-value profile I saw was very different from record-chasing training. Small sample — do not generalize from this.

| Workload | GPU | Location | What I measured | Cost | My note |
|----------|-----|----------|-----------------|------|---------|
| Full BF16 fine-tune of a ~27B parameter open model | H200 SXM | Reykjavik, Iceland | 50-65% GPU util over 6.9 h in my run | ~$28 total for me | Full precision fit in 141 GB without me needing to quantize |
| Quantized inference eval on a ~27B open model | A40 48GB | Montreal, Canada | 27.5 tok/s, GEMM 108.8 TFLOPS on my test | ~$0.47/hr | Community cloud. In my opinion, a good dollars-per-token for quantized inference |
| Quantized inference eval on a ~32B open model | A40 48GB | Montreal, Canada | Queued after the 27B run | ~$0.47/hr | Same pod I kept up, queued sequentially |

My personal takeaway, for my own use: for quantized inference of 27-70B class open models on my workload, A40 community cloud came out cheaper than H100 or H200 in the rentals I ran, and the speed gap felt small compared to the cost gap. For full-precision fine-tunes at the same model size, the VRAM difference mattered more to me than raw TFLOPS. Totally possible your workload works out differently.

## What I ran into on Blackwell GPUs (RTX 5090, DGX Spark GB10)

I'm documenting this because it caught me off guard when I hit it. I am not stating these as general facts — I am describing what happened on the specific pods and dev boxes I used.

- I was able to `pip install -e .` the FA3 (`flash_attn_interface`) package inside the NGC PyTorch container on a Blackwell box, but every `flash_attn_func` call I made failed at runtime with `no kernel image is available for execution on the device`. When I looked at the build log, the kernel object files were all named `sm80` or `sm90` — I didn't see any Blackwell-compiled variants. I reported the pattern on the [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/i-keep-failing-to-install-flash-attention-3-in-the-ltx-2-uv-environment/357560) and on [Dao-AILab/flash-attention#1969](https://github.com/Dao-AILab/flash-attention/issues/1969). Still waiting on a reply. I may be missing something obvious.
- In the same NGC container, `flash_attn` 2.7.4 (FA2) was pre-installed and it ran for me on Blackwell. In the little test I did, FA2 output and `torch.nn.functional.scaled_dot_product_attention` output were identical, so for my workload I swapped to FA2 or SDPA. I'm not claiming this is the right move for everyone.
- The Triton custom kernels in some modded-nanogpt style code I was porting hit `OutOfResources: shared memory` on my Blackwell box because they requested about 180 KB of shared memory per kernel instance, and the card I had reports about 101 KB per SM. Cutting `BLOCK_SIZE_N` and `num_stages` got them launching in my setup. Probably not a universal fix.
- Tooling and runtime support for sm_121 in PyTorch and other libraries was still moving under me while I was using it. See [PyTorch forums on DGX Spark](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744) and [vLLM #31128](https://github.com/vllm-project/vllm/issues/31128) for what people have been reporting. I am not the right person to summarize the current state.

## What I actually spent

On Parameter Golf alone, I rented something like 13 pods across 5 regions and the total came out to roughly $360 out of my own pocket. Breakdown from my own records:

| Segment | My approximate spend | What I got |
|---------|----------------------|-----------|
| Iceland 8xH100 experiments | ~$65 | 3 hours, 10 experiments |
| KC 1xH100 baseline testing | ~$12 | PR #406 plus T1-T5 |
| KC 8xH100 secondary runs | ~$55 | Re-verification plus 3-seed runs |
| March 23-26 early pods (US) | ~$40 | Mostly pods I'd now benchmark and probably kill sooner |
| RTX 5090 testing (Iceland) | ~$6 | 1.5 hours of diagnostics |
| Compliance-adjacent H200 training (Iceland) | ~$28 | Full-precision fine-tune of a ~27B model |
| Quantized inference on A40 (Montreal) | ~$1.20 | 2.5 hours of eval |
| Miscellaneous overnight pods and failures | ~$150 | The really slow pods I wish I'd killed sooner |

A 30-second benchmark could have saved me a meaningful fraction of that last row, is how I read it.

## What's worked for me (not advice)

These are habits from my own rentals. Not recommendations.

1. **I've tried Iceland for H100 SXM work.** In the small number of rentals I did, the Icelandic pods posted the fastest GEMM numbers. I can't explain it and I won't guarantee it for you.
2. **I run the benchmark before starting a long run.** It's saved me money in my own experience. 30 seconds up front has more than once cost me less than one wasted hour later.
3. **On multi-GPU pods I spot-check every GPU.** In one of my rentals, 7 of 8 GPUs looked fine and one was noticeably slower.
4. **If the benchmark looks bad to me, I stop the pod.** RunPod charges by the minute in my experience, so I'd rather eat $0.10 to retry than $40 of slow training.
5. **For my own 8xH100 training work, Iceland has been my first choice.** In the small sample I ran, the combination of GEMM speed, NVLink, and sustained clocks was what I wanted. Not claiming that generalizes.
6. **For my own inference of quantized 27-70B open models, A40 community cloud has been a good fit.** Again, for my workload, my setup, my latency tolerance.
7. **I save the benchmark output.** One command:
   ```bash
   bash pod-test.sh | tee benchmark-$(date +%Y%m%d-%H%M%S).txt
   ```
8. **I use RunPod network volumes for anything I want to survive pod termination.** Container storage on the pods I've rented has not persisted after termination in my experience.
9. **If my code imports FA3, I check the target GPU's compute capability first.** In my runs, FA3 did not work on Blackwell SM 12.x. FA2 or SDPA did. Your toolchain may work differently.

## Finding a specific GPU again

If a pod I rented turned out to be exceptional in my runs, the updated script captures everything I personally need to try to request the same hardware back. It writes two things:

1. A **GPU FINGERPRINT** block to stdout with pod id, datacenter, per-GPU UUID and serial, and the GEMM and memory bandwidth numbers I measured.
2. A **JSON sidecar** (default `/tmp/pod-benchmark-<timestamp>.json`, override with `JSON_OUT=path.json`) containing the same data in a machine-readable form.

I save these alongside my training logs so I can cross-reference later.

### What the script captures

```
pod_id=<runpod pod id>
pod_dc_id=<datacenter>
pod_public_ip=<ip:port>
pod_location=<city, region, country>
gemm_tflops=<my measurement>
mem_bw_gbs=<my measurement>
gpu[0]: name="NVIDIA H100 80GB HBM3" serial=<13-digit> uuid=GPU-<hex uuid> pci=<bus id>
gpu[1]: ...
```

In my understanding, UUID is unique to the physical GPU. Two of my rentals with the same UUID in the log would mean I got the same card back. I have not confirmed this with RunPod directly.

### My workflow for trying to re-rent a specific pod

RunPod does not seem to let me request a specific GPU by UUID directly, at least with the CLI I have. What I do instead:

1. From a saved fingerprint, I note the `pod_dc_id` and the GPU model that performed well for me.
2. I rent a pod in that same datacenter with the same GPU type using the RunPod CLI:
   ```bash
   runpodctl pod create \
     --name "chasing-good-gpu" \
     --gpu "NVIDIA H100 80GB HBM3" \
     --gpuCount 8 \
     --dataCenterId "EUR-IS-1" \
     --templateId "<my PyTorch template>"
   ```
3. I run this benchmark on the new pod.
4. If the new UUID matches a previously saved fast one, I keep the pod. If performance matches but UUID doesn't, it's a different card of the same class (usually fine for me). If the new one is slow, I stop it and retry.

Cost per attempt in my rentals has been roughly $0.10/minute for 1xH100 and $0.35/minute for 8xH100. A few retries to land on a specific card has been cheaper than an hour on a slow one for me.

### Commands I use to inspect RunPod inventory

```bash
runpodctl dc list                              # datacenters, human-readable
runpodctl dc list -o json                      # machine-readable
runpodctl gpu list --include-unavailable       # what GPU types exist and where
```

The `dc_id` from my JSON sidecar maps to the `--dataCenterId` flag.

### My personal known-good-pods habit

Once I had a handful of sidecar files:

```bash
ls -1 /tmp/pod-benchmark-*.json ~/bench-archive/*.json

# With jq
jq -r '[.timestamp, .runpod.dc_id, .gpus[0].uuid, .gemm.tflops] | @tsv' \
    /tmp/pod-benchmark-*.json 2>/dev/null | sort -k4 -nr

# Without jq
grep -H "uuid\|tflops" /tmp/pod-benchmark-*.json
```

Over time I've seen a few specific UUIDs reappear on re-rentals, and the `dc_id` field has been a useful lever for trying to get them back.

## Sample output (what I see on a good pod)

```
═══════════════════════════════════════════
  RunPod GPU Benchmark
═══════════════════════════════════════════

=== RunPod Pod Identity ===
pod_id=a8f3c2b1e7
pod_hostname=some-hostname
pod_public_ip=213.181.122.175:10750
pod_dc_id=EUR-IS-1
pod_location=Reykjavik, Capital Region, IS

=== GPU Identity ===
0, NVIDIA H100 80GB HBM3, 00000000:1B:00.0, 1234567890123, GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

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
213.181.122.175:10750 — Reykjavik, Capital Region, IS

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

  REFERENCE VALUES that my H100 SXM rentals have printed:
  GEMM: < 0.50 ms (> 275 TFLOPS)
  Memory BW: > 2800 GB/s
  Max GPU clock: 1980 MHz

  On my workload, if GEMM > 0.70 ms or BW < 2000 GB/s,
  I typically stop the pod and try again.

  Sidecar JSON saved above. To find this exact GPU again later,
  grep for its uuid or serial across your saved benchmarks.
═══════════════════════════════════════════
```

## Contributing

If you run this benchmark on other GPU types or in other RunPod regions, I'd be interested to see the sidecar output. Open a PR or issue with what you saw. I'd rather add more data points than defend the current set as correct.

### Things I'd personally like to see more data on

- **B200** (RunPod lists it; I haven't rented one yet)
- **L40S** for training and inference
- **A6000** multi-GPU NVLink behavior
- **Region diversity** beyond US and Iceland (APAC, EU-DE, CA-MTL, etc.)
- **AMD Instinct MI300X / MI355X** runs; this script uses `nvidia-smi` only, so an AMD fork would be welcome

## License

MIT

## Author

**Nathan Maine** ([@NathanMaine on GitHub](https://github.com/NathanMaine))

This repository is a personal project on my own time. It is not affiliated with and does not represent the position of any employer, client, or organization. Everything here is my own observation from pods I personally rented, and my own interpretation of those observations.

Built during the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition. [Issue #396](https://github.com/karpathy/autoresearch/issues/396) on karpathy/autoresearch was filed during this same adventure.

---

### Related personal projects

- **[Parameter Golf Experiment Lab](https://github.com/NathanMaine/parameter-golf-experiment-lab)** — my personal dashboard for the competition
- **[NVIDIA Developer Forum: Flash Attention 3 on DGX Spark](https://forums.developer.nvidia.com/t/i-keep-failing-to-install-flash-attention-3-in-the-ltx-2-uv-environment/357560)** — where I posted the FA3 behavior I observed on Blackwell SM 12.1
