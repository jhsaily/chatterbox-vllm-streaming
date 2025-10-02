from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, AsyncGenerator
import time

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from async_lru import alru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.s3gen.s3gen_exportable import S3GenExportable
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm

import uuid
import torch.multiprocessing as mp
import threading
import asyncio


REPO_ID = "ResembleAI/chatterbox"

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

def s3gen_result_collector(result_queue: mp.Queue, futures, loop):
    while True:
        job_id, result = result_queue.get()
        if job_id is None:
            break
        future = futures.pop(job_id, None)
        if future is not None and not future.done():
            loop.call_soon_threadsafe(future.set_result, result)
    
def s3gen_worker(use_fp16: bool, ckpt_dir: str, target_device: str, job_queue: mp.Queue, result_queue: mp.Queue):

    s3gen = S3Gen(use_fp16=use_fp16)
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen = s3gen.to(device=target_device).eval()


    result_queue.put(None)

    while True:
        job = job_queue.get()

        if job is None:
            break
        
        job_id, job_type, job_arguments = job

        if (job_type == "inference"):
            speech_tokens, s3gen_ref, diffusion_steps = job_arguments

            # Wrap
            #s3gen_exportable = S3GenExportable(s3gen.flow, s3gen.mel2wav, s3gen.device, s3gen_ref).eval().cuda()
            #s3gen_exportable.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
            #s3gen_exportable = s3gen_exportable.to(target_device).eval()
            #scripted = torch.jit.trace(s3gen_exportable, (speech_tokens))
            with torch.inference_mode():
                with torch.no_grad():
                    wav, _ = s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )
                    
                    wav.share_memory_()
                    result_queue.put((job_id, (wav)))
        elif (job_type == "ref_conditionals"):
            wav_fpath, DEC_COND_LEN, ENC_COND_LEN, speech_cond_prompt_len = job_arguments
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
            s3gen_ref_dict = s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:ENC_COND_LEN]], max_len=speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            t3_cond_prompt_tokens = t3_cond_prompt_tokens.detach().cpu().clone()
            s3gen_ref_dict['prompt_token'] = s3gen_ref_dict['prompt_token'].detach().cpu().clone()
            s3gen_ref_dict['prompt_token_len'] = s3gen_ref_dict['prompt_token_len'].detach().cpu().clone()
            s3gen_ref_dict['prompt_feat'] = s3gen_ref_dict['prompt_feat'].detach().cpu().clone()
            s3gen_ref_dict['embedding'] = s3gen_ref_dict['embedding'].detach().cpu().clone()
            result_queue.put((job_id, (ref_16k_wav, s3gen_ref_dict, t3_cond_prompt_tokens)))


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: AsyncLLMEngine, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals, s3gen_job_queue: mp.Queue, s3gen_result_queue: mp.Queue, s3gen_workers, s3gen_futures):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.s3gen_job_queue = s3gen_job_queue
        self.s3gen_result_queue = s3gen_result_queue
        self.s3gen_workers = s3gen_workers
        self.s3gen_futures = s3gen_futures

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,

                   # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                   s3gen_use_fp16: bool = False,
                   s3gen_worker_count: int = 5,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / "t3_cfg.safetensors")

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
        # This rough heuristic gives 1.55GB for the model weights plus 128KB per token.
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        engine_args = AsyncEngineArgs(
            model="./t3-model",
            task="generate",
            tokenizer="EnTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=vllm_memory_percent,
            enforce_eager=not compile,
            max_model_len=max_model_len,
            # max_num_batched_tokens=8192
        )

        t3 = AsyncLLMEngine.from_engine_args(engine_args)

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = None

        ctx = mp.get_context("spawn")
        s3gen_job_queue = ctx.Queue()
        s3gen_result_queue = ctx.Queue()
        s3gen_workers = [ctx.Process(target = s3gen_worker, args = (s3gen_use_fp16, ckpt_dir, target_device, s3gen_job_queue, s3gen_result_queue)) for _ in range(s3gen_worker_count)]

        for worker in s3gen_workers:
            worker.start()

        for _ in range(s3gen_worker_count):
            s3gen_result_queue.get() # wait for the worker to be ready
        
        loop = asyncio.get_running_loop()
        s3gen_futures = {}
        result_collector_thread = threading.Thread(
            target=s3gen_result_collector,
            args=(s3gen_result_queue, s3gen_futures, loop),
            daemon=True
        )
        result_collector_thread.start()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds, s3gen_job_queue=s3gen_job_queue, s3gen_result_queue=s3gen_result_queue, s3gen_workers=s3gen_workers, s3gen_futures=s3gen_futures
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",

                        # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                        s3gen_use_fp16: bool = False,
                        s3gen_worker_count: int = 5,
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, s3gen_use_fp16 = s3gen_use_fp16, s3gen_worker_count=s3gen_worker_count, *args, **kwargs)

    @alru_cache(maxsize=10)
    async def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            job_id = str(uuid.uuid4())
            self.s3gen_futures[job_id] = future
            
            self.s3gen_job_queue.put((job_id, "ref_conditionals", (wav_fpath, self.DEC_COND_LEN, self.ENC_COND_LEN, self.t3_config.speech_cond_prompt_len)))
            (ref_16k_wav, s3gen_ref_dict, t3_cond_prompt_tokens) = await future
            t3_cond_prompt_tokens = t3_cond_prompt_tokens.to("cuda")

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb
        
    async def generate(
        self,
        prompt: str,
        audio_prompt_path: Optional[str] = None,
        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000,
        top_p=0.8,
        repetition_penalty=2.0,
        min_chunk_size: int = 10, # Tokens per chunk
        context_window: int = 50,
        fade_duration: float = 0.02,
        request_id = str(uuid.uuid4()),
        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """
        Test impelementation of streaming within chatterbox-vllm
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        s3gen_ref, cond_emb = await self.get_audio_conditionals(audio_prompt_path)

        async for chunk in self.generate_stream_with_conds(
            prompt = prompt,
            s3gen_ref = s3gen_ref,
            cond_emb = cond_emb,
            diffusion_steps = diffusion_steps,
            exaggeration = exaggeration,
            temperature = temperature,
            max_tokens = max_tokens,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            min_chunk_size = min_chunk_size,
            context_window = context_window,
            fade_duration = fade_duration,
            request_id = request_id,
            *args, **kwargs
        ):
            yield chunk

    async def generate_stream_with_conds(
        self,
        prompt: str,
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000,
        top_p=0.8,
        repetition_penalty=2.0,
        min_chunk_size: int = 10, # Tokens per chunk
        context_window: int = 50,
        fade_duration: float = 0.02,
        request_id = str(uuid.uuid4()),
        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text
        prompt = "[START]" + punc_norm(prompt) + "[STOP]"

        with torch.inference_mode():
            sampling_params = SamplingParams(
                temperature=temperature,

                stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                max_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,

                *args, **kwargs,
            )

            generation_input = {
                "prompt": prompt,
                "multi_modal_data": {
                    "conditionals": [cond_emb],
                }
            }

            previous_token_ids = []
            speech_token_buffer = []
            # mp.set_start_method('spawn', force=True)
            async for output in self.t3.generate(generation_input, sampling_params = sampling_params, request_id = request_id):

                token_ids = output.outputs[0].token_ids

                speech_token_buffer = token_ids[len(previous_token_ids):]

                if (len(speech_token_buffer) < min_chunk_size and output.finished == False):
                    continue

                speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in speech_token_buffer], device="cuda")

                if (speech_tokens.squeeze(0).dim() == 0):
                    continue

                previous_token_ids = token_ids

                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561]

                loop = asyncio.get_running_loop()
                future = loop.create_future()
                job_id = str(uuid.uuid4())
                self.s3gen_futures[job_id] = future
                
                self.s3gen_job_queue.put((job_id, "inference", (speech_tokens, s3gen_ref, diffusion_steps)))
                (wav) = await future

                yield wav.cpu()

            # run torch gc
            torch.cuda.empty_cache()
        
        return
        
    def shutdown(self):
        for _ in self.s3gen_workers:
            self.s3gen_job_queue.put(None)
        for worker in self.s3gen_workers:
            worker.join()
        del self.t3
        torch.cuda.empty_cache()
