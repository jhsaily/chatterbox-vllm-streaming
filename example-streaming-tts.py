#!/usr/bin/env python3

from typing import List
import asyncio
import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS
import time
import uuid
import random


MAX_CHUNK_SIZE = 400 # characters

# Process in batches of X chunks at a time. Tune this based on your GPU memory.
#   15 seems to work for 8GB VRAM
#   40 seems to work for 16GB VRAM
#   80 seems to work for 24GB VRAM
# You may need to adjust the batch size based on your GPU memory.
BATCH_SIZE = 40


async def run_tts(model, prompt: str, test: int):
    streamed_chunks = []
    chunk_count = 0
    cumulative_duration = 0.0

    request_id = str(uuid.uuid4())
    start_time = time.time()
    last_chunk_finish_time = start_time

    async for audio_chunk in model.generate(prompt, audio_prompt_path= "./LJ037-0171.wav",exaggeration=0.5, diffusion_steps=5, min_chunk_size=2, request_id=request_id):
        gen_time = time.time()

        chunk_count += 1
        chunk_gen_time = gen_time - start_time
        time_since_last_chunk = gen_time - last_chunk_finish_time
        last_chunk_finish_time = gen_time
        
        streamed_chunks.append(audio_chunk)
        chunk_duration = audio_chunk.shape[-1] / model.sr
        cumulative_duration += chunk_duration
        
        # ta.save(f"test-{test}-chunk-{chunk_count}.wav", audio_chunk, model.sr)
        print(f"[Chatterbox-vLLM] (Request {request_id}) Time to chunk {chunk_count}: {chunk_gen_time:.2f}s, duration: {chunk_duration:.3f}s, cumulative duration: {cumulative_duration:.3f}s, since last chunk: {time_since_last_chunk:.2f}s")
    
    audio_gen_time = time.time() - start_time
    full_streamed_audio = torch.cat(streamed_chunks, dim=-1)
    audio_duration = full_streamed_audio.shape[-1] / model.sr
    ta.save(f"test-{test}.wav", full_streamed_audio, model.sr)
    print(f"[Chatterbox-vLLM] (Request {request_id}) TTS generation time: {audio_gen_time:.2f}s, Total Duration: {audio_duration:.3f}s")


async def main():
    model = ChatterboxTTS.from_pretrained(
        max_batch_size = BATCH_SIZE,
        max_model_len = MAX_CHUNK_SIZE * 3, # Rough heuristic
        s3gen_use_fp16 = True,
        s3gen_worker_count = 1,
    )

    prompt1 = "Hey buddy, what's going on? Do you need help with something, or are you just browsing?"
    prompt2 = f"You are listening to a demo of the Chatterbox TTS model running on VLLM. This is a simple prompt to test the streaming implementation. And here is another sentence that is a bit longer than the previous one, but not by much."

    await run_tts(model, prompt1, 1),
    await asyncio.gather(
        run_tts(model, prompt2, 2),
    )

    model.shutdown()


if __name__ == "__main__":
    asyncio.run(main())