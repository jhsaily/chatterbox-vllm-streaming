import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class S3GenExportable(nn.Module):
    """
    TorchScript/ONNX-friendly wrapper for S3Gen.
    - Precompute ref_dict outside the model (speaker embeddings, ref mels, tokens).
    - Fix n_timesteps at init (no Python loops depending on runtime args).
    - Inputs are just speech_tokens + ref_dict tensors.
    """

    def __init__(self, flow, mel2wav, device, ref_dict):
        super().__init__()
        self.flow = flow
        self.mel2wav = mel2wav
        self.device = device
        self.ref_dict = ref_dict

    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        n_timesteps: int = 10,
    ):
        assert (ref_wav is None) ^ (self.ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"
        print('hello, does this even wokr?')
        print(self.ref_dict)

        if self.ref_dict is None:
            self.ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(self.ref_dict):
                if isinstance(self.ref_dict[rk], np.ndarray):
                    self.ref_dict[rk] = torch.from_numpy(self.ref_dict[rk])
                if torch.is_tensor(self.ref_dict[rk]):
                    self.ref_dict[rk] = self.ref_dict[rk].to(self.device)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"
        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)
        
        return self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            n_timesteps=n_timesteps,
            **self.ref_dict,
        )

    def hift_inference(self, speech_feat, cache_source: Optional[torch.Tensor] = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(self.device)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        cache_source: Optional[torch.Tensor] = None, # NOTE: this arg is for streaming, it can probably be removed here
        finalize: bool = True,
        no_trim: bool = False,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """
        Args:
            speech_tokens: [B, T] discrete speech token ids
            prompt_token: [B, T_p] reference speech tokens
            prompt_token_len: [B] length of prompt_token
            prompt_feat: [B, T_m, n_mels] reference mel features
            prompt_feat_len: [B] length of prompt_feat (can be dummy if unused)
            embedding: [B, D] speaker embedding
        Returns:
            output_wavs: [B, T_audio]
        """

        # Run diffusion model → mel
        output_mels = self.flow_inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize, n_timesteps=n_timesteps)

        # Run HiFi-GAN vocoder
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        if not no_trim:
            fade_len = min(output_wavs.shape[1], len(self.trim_fade))
            output_wavs[:, :fade_len] *= self.trim_fade[:fade_len]
            #output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources