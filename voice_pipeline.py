"""
Fully offline voice agent pipeline: speech-in -> speech-out.

Chains three local, open-source components:
  1. faster-whisper (STT)      - transcribes an input WAV file
  2. Ollama + a local LLM       - generates a response to the transcribed text
  3. Piper (TTS)                - synthesizes the response back to speech

No cloud APIs, no internet required after the models are downloaded once.

Usage:
    python3 voice_pipeline.py --input command.wav --output response.wav

Requirements:
    pip install faster-whisper requests
    pip install piper-tts
    ollama pull qwen3-vl:2b-instruct   (or any instruct-tuned model you prefer)

Tested on:
    - Raspberry Pi 5 (16GB, CPU-only)
    - NVIDIA Jetson Orin Nano Developer Kit (8GB, CUDA-accelerated via Ollama)

See the accompanying README for full benchmark results on both boards.
"""

import argparse
import subprocess
import time

import requests
from faster_whisper import WhisperModel

DEFAULT_MODEL = "qwen3-vl:2b-instruct"
DEFAULT_VOICE = "en_US-lessac-medium.onnx"
OLLAMA_URL = "http://localhost:11434/api/generate"


def run_pipeline(audio_in: str, audio_out: str, voice_model_path: str, llm_model: str) -> dict:
    pipeline_start = time.time()
    timings = {}

    # Stage 1: Speech-to-text
    t0 = time.time()
    stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = stt_model.transcribe(audio_in, beam_size=1)
    transcribed_text = " ".join(seg.text for seg in segments).strip()
    timings["stt"] = time.time() - t0
    print(f"[STT] {timings['stt']:.2f}s -> \"{transcribed_text}\"")

    # Stage 2: LLM response (via local Ollama API)
    t0 = time.time()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": llm_model,
            "prompt": f"Respond briefly and naturally to: {transcribed_text}",
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    llm_text = response.json()["response"].strip()
    timings["llm"] = time.time() - t0
    print(f"[LLM] {timings['llm']:.2f}s -> \"{llm_text}\"")

    # Stage 3: Text-to-speech
    t0 = time.time()
    subprocess.run(
        f'echo "{llm_text}" | python3 -m piper -m {voice_model_path} -f {audio_out}',
        shell=True,
        capture_output=True,
        check=False,
    )
    timings["tts"] = time.time() - t0
    print(f"[TTS] {timings['tts']:.2f}s -> saved to {audio_out}")

    timings["total"] = time.time() - pipeline_start
    print(f"\n--- TOTAL PIPELINE TIME: {timings['total']:.2f}s ---")
    print(f"STT: {timings['stt']:.2f}s | LLM: {timings['llm']:.2f}s | TTS: {timings['tts']:.2f}s")

    return {
        "transcribed_text": transcribed_text,
        "llm_text": llm_text,
        "timings": timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline speech-in -> speech-out benchmark pipeline")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--output", default="response.wav", help="Path to write the synthesized response")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Path to a Piper .onnx voice model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name to use for the response")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.voice, args.model)