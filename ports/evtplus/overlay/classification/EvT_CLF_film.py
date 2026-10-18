"""EvT_CLF_film.py -- FiLM-Rate conditioning applied to EvT+ (attention paradigm).

Fourth architecture family in the cross-backbone robustness study (after
PointNet++, EventMamba, AEGNN): a Perceiver-style transformer that loops
over T event-frame time-steps, using cross-attention to update a small set
of latent memory vectors. FiLM injection points, mirroring the SA1-3
granularity used elsewhere:
  - after the per-timestep self-attention backbone (self.backbone_events),
    on the token features x, before they update the memory -- analogous to
    conditioning "SA1..SA3" per set-abstraction stage.
  - on the final pooled latent memory (self.cross_attention.prev_latent_vectors)
    right before the decoder -- analogous to conditioning the pooled
    descriptor f_E.
Zero-initialized so FiLM = identity at init, matching the paper's convention.
"""
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from EvT_CLF import LatentMemory, CLF_Decoder
from model_utils.evt_utils import get_block, fourier_features
import copy
import math


class FiLMGenerator(nn.Module):
    def __init__(self, dims, hidden=32):
        super().__init__()
        self.dims = dims
        total_out = 2 * sum(dims)
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, total_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        out = self.net(c.unsqueeze(-1))  # [B, total_out]
        chunks = torch.split(out, [2 * d for d in self.dims], dim=-1)
        return [torch.split(ch, [d, d], dim=-1) for d, ch in zip(self.dims, chunks)]


class EvT_CLF_FiLM(nn.Module):
    def __init__(self,
                 embed_dim, events_dim,
                 downsample_pos_enc,
                 pos_encoding, pos_enc_grad,
                 token_projection,
                 preproc_tokens,
                 skip_conn_backbone,
                 init_self_attention_params,
                 latent_memory_params,
                 decoder_params,
                 ):
        super().__init__()

        self.downsample_pos_enc = downsample_pos_enc
        self.embed_dim = embed_dim
        self.skip_conn_backbone = skip_conn_backbone

        pos_enc_params = copy.deepcopy(pos_encoding['params'])
        pos_enc_params['shape'] = (math.ceil(pos_encoding['params']['shape'][0] / downsample_pos_enc),
                                    math.ceil(pos_encoding['params']['shape'][1] / downsample_pos_enc))
        self.pos_encoding = nn.Parameter(fourier_features(**pos_enc_params).permute(1, 2, 0), requires_grad=pos_enc_grad)
        pos_emb_dim = self.pos_encoding.shape[2]

        token_projection['params']['embed_dim'] = embed_dim
        preproc_tokens['params']['embed_dim'] = embed_dim

        self.projection_events = get_block(token_projection['name'], {**token_projection['params'], **{'ipt_dim': events_dim}})
        inp_dim = int(token_projection['params']['init_layers'][-1].split('_')[1])
        if inp_dim == -1:
            inp_dim = embed_dim
        self.preproc_block_events = get_block(preproc_tokens['name'], {**preproc_tokens['params'], **{'ipt_dim': inp_dim + pos_emb_dim}})

        init_self_attention_params['params']['opt_dim'] = embed_dim
        init_self_attention_params['params']['cross_att'] = False
        init_self_attention_params['params']['latent_blocks'] = init_self_attention_params['num_blocks'] - 1
        self.backbone_events = get_block(init_self_attention_params['name'], init_self_attention_params['params'])
        del init_self_attention_params['params']['cross_att']

        latent_memory_params['attention_params']['opt_dim'] = embed_dim
        self.cross_attention = LatentMemory(**latent_memory_params)

        decoder_params['opt_dim'] = embed_dim
        self.decoder = CLF_Decoder(**decoder_params)

        # FiLM stage 1: per-timestep token features (dim=embed_dim)
        # FiLM stage 2: pooled latent memory (dim=embed_dim), right before decoder
        self.film_generator = FiLMGenerator([embed_dim, embed_dim])

    def forward(self, event_data, c: torch.Tensor, init_memory=True):
        # c: [B] centered log-rate
        (g1, b1), (g2, b2) = self.film_generator(c)

        tokens_events, pixels_events = event_data['event_frames']
        pixels_events = torch.div(pixels_events, self.downsample_pos_enc, rounding_mode='trunc')
        B, T, _, _ = tokens_events.shape
        tokens_events = tokens_events.permute(1, 0, 2, 3)
        pixels_events = pixels_events.permute(1, 0, 2, 3)
        samples_mask_events = tokens_events.sum(-1) == 0
        samples_mask_time_events = tokens_events.sum(-1).sum(-1) == 0

        tokens_events_1 = self.projection_events(tokens_events)
        pos_embs = self.pos_encoding[pixels_events[:, :, :, 1], pixels_events[:, :, :, 0], :]
        tokens_events_2 = torch.cat([tokens_events_1, pos_embs], dim=-1)
        tokens_events_2 = self.preproc_block_events(tokens_events_2)
        tokens_events = tokens_events_2

        if init_memory:
            self.cross_attention.init_forward_pass(B)

        for num_time_step in range(T):
            mask_t = samples_mask_events[num_time_step]
            mask_time_t = samples_mask_time_events[num_time_step]
            shortcut = tokens_events[num_time_step].permute(1, 0, 2)  # (tokens, B, emb_dim)

            if B == 1:
                shortcut = shortcut[~mask_t[0]]
                mask_t = mask_t[:, ~mask_t[0]]
                if shortcut.shape[0] <= 1 or mask_time_t.all():
                    continue

            x = self.backbone_events(shortcut, mask=mask_t, mask_time_t=mask_time_t)[0]
            if self.skip_conn_backbone:
                x = x + shortcut
            # ---- FiLM stage 1: modulate per-timestep token features ----
            # x: (tokens, B, emb_dim); gamma/beta: (B, emb_dim) -> broadcast over tokens
            x = (1 + g1.unsqueeze(0)) * x + b1.unsqueeze(0)

            self.cross_attention.update_latent_vectors_from_tokens(x, mask_t=mask_t, mask_time_t=mask_time_t)

        # ---- FiLM stage 2: modulate pooled latent memory before decoding ----
        z = self.cross_attention.prev_latent_vectors  # (num_latent, B, emb_dim)
        z = (1 + g2.unsqueeze(0)) * z + b2.unsqueeze(0)
        return self.decoder(z)
