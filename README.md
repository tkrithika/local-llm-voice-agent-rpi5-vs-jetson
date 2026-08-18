![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Jetson](https://img.shields.io/badge/Jetson%20Orin%20Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

# local-llm-voice-agent-rpi5-vs-jetson

A hands-on benchmark of running a local LLM/VLM and a fully offline voice agent pipeline (STT → LLM → TTS) on two popular low-cost edge boards: Raspberry Pi 5 and Jetson Orin Nano. No cloud, no discrete GPU required (well, mostly).

**TL;DR:** Both boards can run a small vision-language model (Qwen3-VL-2B) and a complete offline voice pipeline entirely locally. The Jetson's GPU gives it a real, sometimes dramatic, speed advantage, but only 8GB of shared CPU/GPU memory means it's far more fragile under real-world conditions than the CPU-only, 16GB Raspberry Pi 5. Full results, including the memory-crash debugging saga, below.

---

## Repo structure

```
local-llm-voice-agent-rpi5-vs-jetson/
├── README.md              # this file: full writeup and benchmark results
├── voice_pipeline.py       # runnable STT -> LLM -> TTS pipeline script
├── requirements.txt        # Python dependencies
└── samples/
    ├── command_test_input.wav        # example input: a spoken voice command
    └── pipeline_output_2.wav   # example output: the pipeline's spoken response
```

To reproduce this benchmark yourself, see [Running it yourself](#running-it-yourself) below.

---

## Setup

| | Raspberry Pi 5 | Jetson Orin Nano Developer Kit |
|---|---|---|
| RAM | 16GB | 8GB (unified CPU/GPU memory) |
| Compute | 4x Cortex-A76 CPU, no GPU | 4x Cortex-A76 CPU + Ampere GPU (CUDA 8.7) |
| Serving layer | Ollama v0.32.9, CPU-only | Ollama v0.32.11, GPU (CUDA) auto-detected |
| Model | `qwen3-vl:2b` and `qwen3-vl:2b-instruct` (Apache 2.0) | Same |
| STT | faster-whisper (Whisper Tiny, int8) | Same |
| TTS | Piper (`en_US-lessac-medium`) | Same |

Both boards were tested with the same prompts, the same test image (a generic dashboard/map screenshot, ~1900x1000px), and the same voice pipeline script (`voice_pipeline.py` in this repo).

---

## Part 1: Text-Only Latency

Qwen3-VL ships in two flavors: a "thinking" variant that generates a visible chain-of-thought before answering, and an "instruct" variant that answers directly. This distinction matters a lot in practice.

| Variant | RPi5 | Jetson (warm) |
|---|---|---|
| Thinking | 35.6s | 18.4s |
| Instruct | 3.7s – 14.0s | 8.78s |

**Takeaway:** the "thinking" variant is 3-4x slower than "instruct" for the exact same question, on both boards. It's not a hardware issue; it's a model-behavior issue. If you're building anything latency-sensitive, use the instruct variant. No prompt flag (`/no_think`, API `think:false`) could suppress the thinking trace on the base variant; you have to switch models entirely.

The Jetson's GPU gave roughly a 2x speedup on the thinking variant once warm, but essentially no speedup on the instruct variant. Short text generation was never heavily compute-bound to begin with, so GPU acceleration had little to work with.

---

## Part 2: Multimodal (Image) Latency and a Memory Crash Investigation

This is where the story gets interesting.

### RPi5: works at every resolution, just slowly

| Image size | Total latency | Image processing time |
|---|---|---|
| Full-size (1906×1057) | 3m 55.9s | 173.5s |
| 800×800 | 2m 16.1s | 64.4s |
| 400×400 | 1m 51.6s | 64.8s (no improvement below 800px: the vision encoder hits a fixed token budget) |

**Finding:** image resolution is the dominant latency lever, not the LLM itself. But there's a floor: shrinking below ~800px on the long edge gives no further speed benefit, only worse accuracy. 800px is the sweet spot.

### Jetson: fast, but only if nothing else is competing for memory

The identical image test **crashed** on the Jetson, at every resolution, with:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 322.49 MiB on device 0: cudaMalloc failed: out of memory
```

The LLM itself loaded fine (~1GB), but the vision encoder couldn't get the extra ~322MB it needed. Digging into `journalctl -u ollama` logs confirmed this was happening right at the vision-encoding step (`process_mtmd`), not during text generation.

**Root cause #1:** an unrelated Docker Compose stack (Grafana + InfluxDB + Mosquitto + a simulator container) was running in the background from earlier project work, quietly eating memory the whole time.

```bash
docker stop $(docker ps -q)
```

Freed ~300MB, just enough. The 800×800 image test then succeeded:

| Run | Total | Image processing |
|---|---|---|
| Cold | 30.57s | 4.12s (mostly one-time init) |
| Warm | **17.85s** | **0.07s** |

That's roughly a **900x speedup** on the image-encoding step alone, once warm and once memory was freed, a dramatic demonstration of what the GPU can actually do when it has room to work.

**But the full-size image (2000 tokens vs. 1052 at 800×800) never succeeded, at any memory state.** This is a hard, resolution-dependent ceiling on this 8GB board, not something freeing more memory fixes.

**Root cause #2, discovered later:** reconnecting to the board's desktop via VNC (to grab screenshots) caused the *exact same* 800×800 request to fail again with the same OOM error, even with Docker still stopped.

| System state | Available memory | 800×800 result |
|---|---|---|
| SSH-only, no desktop, Docker stopped | 6.0–6.3 GiB | ✅ Success |
| VNC desktop session active | 5.6 GiB | ❌ Out of memory |

**Takeaway:** on an 8GB unified-memory board, a desktop environment (X server, display manager, VNC server) is a real, measurable competitor for the same memory your model needs. Development convenience and inference reliability were directly in tension here. Worth noting: this is very likely a **development-kit artifact**, not a hardware ceiling; a production deployment running headless (no desktop, no unrelated services) on the same silicon would likely have meaningfully more headroom.

---

## Part 3: Voice Agent Pipeline (Speech-In → Speech-Out)

Both boards ran the same fully offline pipeline: faster-whisper for STT, Qwen3-VL-2B-instruct for the response, Piper for TTS. Sample input/output audio from this test is in [`samples/`](./samples).

| Board | STT | LLM | TTS | **Total** |
|---|---|---|---|---|
| RPi5 (JFK clip) | 2.69s | 10.48s | 3.46s | **16.62s** |
| Jetson (best run) | 2.84s | 2.97s | 3.19s | **8.99s** |
| Jetson (observed range across repeats) | 2.84s – 17.74s | 2.01s – 2.97s | 2.60s – 3.19s | 8.99s – 23.15s |

**Findings:**
- The **LLM stage was consistently 3-5x faster on the Jetson**: confirmed with `tegrastats` showing the GPU hitting 99% utilization during generation. This is the clearest, most reliable GPU win in the whole benchmark.
- **STT timing on the Jetson was surprisingly unstable**: identical input, identical cached model, but timings ranged from 2.84s to 17.74s across repeated runs. `tegrastats` traces showed CPU cores oscillating between 729MHz and 1497MHz during the slow runs: consistent with CPU frequency-scaling (governor) behavior catching a core "cold" right when the request landed. The RPi5 showed no equivalent instability across any of its own repeated tests.
- Simple, direct voice commands ("turn off the living room light") consistently produced much faster LLM responses than open-ended or reflective input, on both boards, a clear, actionable prompt-design lesson: keep responses short and direct if latency matters.

---

## Overall Takeaways

1. **The Jetson's GPU delivers real, sometimes dramatic speedups**: up to ~900x on image encoding, ~5x on LLM generation, when it has enough free memory to work with.
2. **8GB of shared CPU/GPU memory is a real operational constraint**, not just a number on a spec sheet. Background services and even a remote desktop session were each independently enough to push multimodal inference into failure.
3. **The Raspberry Pi 5, despite having no GPU at all, was the more predictable board**: every test produced stable, repeatable numbers. Slower on average, but you always knew what you were getting.
4. **Model choice matters as much as hardware**: switching from the "thinking" to the "instruct" variant of the same model cut latency by 3-4x, no hardware changes required.
5. **Resize your images.** On both boards, image resolution was either the dominant latency cost (RPi5) or the difference between working and crashing entirely (Jetson).

---

## Running it yourself

**Prerequisites (both boards):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:2b-instruct

pip install -r requirements.txt
# faster-whisper, piper-tts, and requests: see requirements.txt

# Download a Piper voice model, e.g.:
mkdir -p piper_voices && cd piper_voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

**Run the voice pipeline:**
```bash
python3 voice_pipeline.py --input samples/command_test.wav --output response.wav --voice piper_voices/en_US-lessac-medium.onnx
```

**For the image/multimodal test:** resize your test image to ~800px on the long edge first (see Part 2 above for why), then send it to Ollama's `/api/generate` endpoint with the `images` field set to the base64-encoded file. Not included as a script here since it's a single curl call; see Part 2 for the exact request format.

**If you hit a `cudaMalloc failed: out of memory` error on Jetson:** check for other memory-hungry processes first (`docker ps`, and whether a desktop/VNC session is active) before assuming the model itself won't fit; see the debugging notes in Part 2.

---

## Tools used

- [Ollama](https://ollama.com): local model serving
- [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF): small vision-language model (Apache 2.0)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): speech-to-text
- [Piper](https://github.com/rhasspy/piper): text-to-speech
- `tegrastats`: Jetson GPU/CPU/power monitoring
- ImageMagick: image resizing for the resolution comparison

Related write-ups: 

[headless VNC setup guide](https://github.com/tkrithika/jetson-orin-nano-headless-vnc-setup) and [CUDA build guide](https://github.com/tkrithika/jetson-opencv-cuda-build-guide) for the Jetson Orin Nano, written while setting up this benchmark. Also see the [Dockerized IoT stack](https://github.com/tkrithika/jetson-docker-iot-stack) that turned out to be the background process behind the memory-crash investigation in Part 2.

---

## Things worth flagging (not fully covered here)

- **Live microphone input was not tested**: all voice pipeline testing used pre-recorded WAV files. Real-time audio capture and wake-word detection remain untested.
- **Only one model family (Qwen3-VL) was benchmarked.** Other small VLMs (Ministral 3, SmolVLM2, Gemma 3n) may behave differently, especially under the Jetson's memory constraints.
- **Only Ollama was tested as a serving layer.** A direct llama.cpp comparison might behave differently under memory pressure, since Ollama adds its own overhead on top of the underlying runtime.
- **Single hardware units, single test runs per configuration (mostly).** Numbers (especially the Jetson's STT variability) would benefit from more repeated trials to build a proper distribution rather than a handful of sample points.
- **The Jetson findings reflect a developer kit running a full desktop environment**, not a headless production deployment. A stripped-down, headless setup on the same silicon would likely show less memory pressure than reported here.

---

## License

MIT. See [LICENSE](./LICENSE).

---

*All testing was done using only open-source tools and publicly available models, on hardware accessible to the author. Numbers are from a small number of runs on specific hardware units and should be treated as indicative, not definitive. Your mileage will vary with different board revisions, JetPack/OS versions, and thermal conditions.*
