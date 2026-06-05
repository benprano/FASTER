from typing import List
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ATRUCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, seq_len, num_heads=4):
        super(ATRUCell, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.register_buffer("factor", torch.FloatTensor([0.5]))
        self.register_buffer('c1_const', torch.Tensor([1]).float())
        self.register_buffer("factor_impu", torch.FloatTensor([0.5]))
        self.register_buffer('c2_const', torch.Tensor([np.e]).float())
        self.register_buffer("imp_weight_freq", torch.FloatTensor([0.05]))
        self.register_buffer("Wdelta", torch.ones([self.input_size, 1, 1]).float())
        self.register_buffer("features_decay", torch.Tensor(torch.arange(self.input_size)).float())
        self.register_buffer("ones_const", torch.ones([self.input_size, 1, self.hidden_size]).float())
        self.register_buffer("fixed_decay", torch.arange(self.input_size).float())
        # Learnable scaling per feature to modulate decay based on frequency importance
        self.freq_weight = nn.Parameter(torch.zeros(self.input_size))
        self.freq_bias = nn.Parameter(torch.full((self.input_size,), 0.5413))
        self.elastic_slope = nn.Parameter(torch.zeros(1))

        # ── New learnable parameters (register alongside existing ones) ─────────────
        # Per-feature decay modulation (replaces scalar elastic_slope)
        self.feat_slope = nn.Parameter(torch.zeros(self.input_size))  # per-feature offset
        self.sparsity_gate = nn.Parameter(torch.zeros(1))  # learned sparsity sensitivity
        # Soft attention over decay (selects which time-steps matter most)
        self.decay_attn = nn.Parameter(torch.ones(self.input_size))  # feature-wise attention gain

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))

        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        # Interpolation
        self.W_ht_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_h_mem = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_in_imp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size * 3,
                                                                                     self.hidden_size)))
        self.W_ht_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_freq_imp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(1, self.input_size)))
        self.gate_alpha = nn.Parameter(torch.ones(1, self.input_size) * 0.1)
        self.b_freq = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(1, self.input_size)))
        # Semantic salience Gate
        self.U_s = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_gamma = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.W_s = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_gamma = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.b_s = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.U_cc = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        # Query, Key, Value projections for feature correlation

        self.W_q = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_k = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_v = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.log_tau = nn.Parameter(torch.tensor([0.0]))
        
        # Diagonal SSM: A = diag(exp(-softplus(lambda)))
        self.log_lambda = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.hidden_size,)))
        self.x_proj = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size * 2, 1,self.hidden_size * 3)))
        self.x_proj_back = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                        size=(self.input_size * 2, self.input_size, self.hidden_size * 3)))
        self.latent_prior = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                     size=(1, 1, self.hidden_size)))

        # We project from hidden_size to hidden_size
        self.time_weight = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                     size=(self.hidden_size, self.hidden_size)))
        # Optional: Add a bias if you want to fully replicate nn.Linear behavior
        self.time_bias = nn.Parameter(torch.zeros(self.hidden_size))


    @torch.jit.script_method
    def forward(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x, t, l, freq = inputs, times, last_data, freq_list
        # Apply temporal decay to D-STM
        delta_time = self.time_decay_encode(t)
        decay_factor = torch.mul(delta_time, self.freq_decaying(freq, h_tilda_t))
        # Trace-Semantic Memory with Temporal Attention
        c_tilda_t, alpha_scores = self.tsmta(decay_factor.permute(0, 2, 1), c_tilda_t, delta_time, h_tilda_t)
        # frequency weights for imputation of missing data based on frequencies of features
        time_enc = self.time_decay_encode(t)  # [B, T, H]
        freq_enc = self.freq_encode(freq)  # [B, F, H]
        h_mod = self.modulate_hidden(h_tilda_t, freq_enc, time_enc)
        c_mod = self.modulate_hidden(c_tilda_t, freq_enc, time_enc)
        x_last_hidden = torch.tanh(torch.einsum("bij,ijk->bik", h_mod, self.W_ht_last) +
                                   torch.einsum("bij,ijk->bik", c_mod, self.W_ct_last) +
                                   self.b_j_last).permute(0, 2, 1)

        imputat_imputs = torch.tanh(torch.einsum("bij,ijk->bik", h_mod, self.W_ht_mask) +
                                    torch.einsum("bij,ijk->bik", c_mod, self.W_ct_mask) +
                                    self.b_j_mask).permute(0, 2, 1)
        # Replace nan data with the impuated value generated from LSTM memory and frequencies weights
        # frequencies, decay, attn_scores, attended, attn_weights, imputed_missed_x, x_imputed
        _,_,_,_,_,_,_,_, x_last = self.facit(l, freq, t, x_last_hidden)
        (freq_facit, decay_slope, attn_scores,interpolated_hidden, adaptive_factor, attended_cross,
         att_weights_cross, all_imputed_x, imputed_x) = self.facit(x, freq, t, imputat_imputs)
        # Ajust previous to incoporate the latest records for each feature
        last_tilda_t = F.elu(torch.einsum("bij,jik->bjk", x_last, self.U_last) + self.b_last)
        h_tilda_t = h_tilda_t + last_tilda_t
        # Input Gate
        i_t = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_i) +
                            torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                            c_tilda_t * self.W_cell_i)
        # gamma_t
        gamma_t = self.time_decay_encode(t)

        # Candidate Memory Cell
        c = torch.tanh(torch.einsum("bij,jik->bjk", imputed_x, self.U_c) +
                       torch.einsum("bij,ijk->bik", gamma_t, self.W_c) + self.b_c)


        # Capturing Temporal Dependencies wrt to the previous hidden state attended, att_weights
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                               torch.einsum("bij,jik->bjk", imputed_x, self.U_j) +
                               self.b_j)
        h_tilda_t = (1 - i_t) * h_tilda_t + i_t * c * j_tilda_t

        return (h_tilda_t, c, attn_scores,
                decay_factor, attended_cross, alpha_scores,
                att_weights_cross, freq_facit, decay_slope,
                interpolated_hidden,adaptive_factor, all_imputed_x)
   
    @torch.jit.script_method
    def facit(self, x: torch.Tensor, freq_dict: torch.Tensor, delta_t: torch.Tensor,
              x_hidden: torch.Tensor):
        # x_hidden shape: (bs, hidden_size, nb_features)
        bs, h_dim, n_feat = x_hidden.shape
        mask = torch.isnan(x)
        # Step 1: Frequency-based scaling to account for timing irregularity
        # Step 1-a) Frequency-driven feature factor
        freqs = (self.seq_len - freq_dict)
        factor_feature = torch.div(torch.exp(-self.imp_weight_freq * (freq_dict + self.b_freq)),
                                   torch.exp(-self.imp_weight_freq * (freq_dict + self.b_freq)).max()
                                   ).unsqueeze(1)

        # Step 1-b) Frequency-driven imputation factor
        factor_imp = torch.div(torch.exp(self.factor_impu * (freq_dict + self.b_freq_imp)),
                               torch.exp(self.factor_impu * (freq_dict + self.b_freq_imp)).max()
                               ).unsqueeze(1)

        # Step 1-c) Adjusted frequencies
        # ── Step 2: Sparsity-aware global slope ──
        # sparsity_ratio ∈ [0, 1]: fraction of MISSING values
        sparsity_ratio = mask.float().mean()  # scalar

        # Learned sensitivity to sparsity (sigmoid gates to [0, 1])
        sparsity_sensitivity = torch.sigmoid(self.sparsity_gate)  # ∈ (0, 1)

        # Global slope: [0.5, 1.5] base + learned sparsity boost
        global_slope = (
                0.5 + torch.sigmoid(self.elastic_slope)  # ∈ [0.5, 1.5]
                + sparsity_sensitivity * (1.0 - sparsity_ratio)  # learned boost
        )

        # ── Step 3: Per-feature slope correction ──
        # feat_slope is initialized to 0 → starts identical to Mechanism 1
        # softplus ensures a strictly positive per-feature contribution
        feat_correction = F.softplus(self.feat_slope)  # shape (n_feat,)
        combined_slope = global_slope * feat_correction  # (n_feat,)

        # ── Step 4: Structured decay using fixed positional index ───────────────
        exponent = (
                -self.factor
                * combined_slope  # (1, n_feat)
                * self.fixed_decay  # (seq_len, 1)
        )  # → (seq_len, n_feat)
        decay = torch.exp(exponent)  # (seq_len, n_feat)
        # ── Step 5: Soft decay attention gate ───────────────────────────────────
        # Learns to amplify or suppress decay per-feature beyond the slope
        # sigmoid keeps gate in (0, 1); initialized to 1 → neutral start
        attn_gate = torch.sigmoid(self.decay_attn)  # (n_feat,)
        decay = decay * attn_gate  # (seq_len, n_feat)

        # ── Step 6: Apply to adjusted frequencies ───────────────────────────────
        frequencies = freqs * decay
        frequencies = torch.div(frequencies, frequencies.max()).unsqueeze(-1)

        # Step 2: Temporal attention from recent history
        attn_scores = torch.softmax(torch.einsum("bij,ijk->bik", x_hidden.permute(0, 2, 1), self.W_h_mem), dim=1)
        dynamic_context = torch.sum(attn_scores.permute(0, 2, 1) * x_hidden, dim=1).unsqueeze(1)
        # Step 3: Dynamic interpolation between high-confidence and semantic-smoothed
        smoothed_hidden = dynamic_context * x_hidden + (1 - dynamic_context) * torch.tanh(x_hidden)
        # Step 4: Adaptive gating based on delta_t and context
        delta_gate = self.time_decay_encode(delta_t).permute(0, 2, 1)  # [B, 1, F]
        # Cross-Attention to learn feature correlations
        # --- Forward Pass ---
        # Generate Q, K, V with latent richness
        Q = torch.einsum("bnf,nfd->bnd", x_hidden.permute(0, 2, 1), self.W_q)  # (bs, n_feat, h_dim)
        K = torch.einsum("bnf,nfd->bnd", x_hidden.permute(0, 2, 1), self.W_k)
        V = torch.einsum("bnf,nfd->bnd", x_hidden.permute(0, 2, 1), self.W_v)

        # Split into Heads: (bs, heads, n_feat, head_dim)
        Q = Q.view(bs, n_feat, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, n_feat, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, n_feat, self.num_heads, self.head_dim).transpose(1, 2)
        # 4. Compute Feature-to-Feature Attention Logits
        attn_logits = torch.matmul(Q, K.transpose(-1, -2))  / math.sqrt(self.head_dim)
        """
        tau = torch.exp(self.log_tau)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) * tau / math.sqrt(self.head_dim)
        """
        # Apply mask to the feature-to-feature matrix
        attn_logits = attn_logits.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 1, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # Compute attended values: (bs, heads, n_feat, head_dim)
        attended = torch.matmul(attn_weights, V)
        # Reshape back: (bs, n_feat, h_dim) bs, h_dim, n_feat
        attended = attended.transpose(1, 2).contiguous().view(bs, h_dim, n_feat)
        # residual learning
        attended = attended + x_hidden
        # assert attn_weights.std() > 1e-4, "Attention still collapsed"
        adaptive_factor = 0.5 * (delta_gate + dynamic_context + attended)
        # Step 5: Dynamic interpolation between high-confidence and semantic-smoothed
        interpolated_hidden = adaptive_factor * smoothed_hidden + (1 - adaptive_factor) * x_hidden
        # Step 6: Compute imputed values
        imputed_missed_x = (
                factor_imp * frequencies.permute(0, 2, 1) * interpolated_hidden +
                (1 - factor_imp) * factor_feature * interpolated_hidden
        )
        # Step 7: Replace missing values
        x_imputed = torch.where(torch.isnan(x.unsqueeze(1)), imputed_missed_x, x.unsqueeze(1))
        return (frequencies.squeeze(-1), decay, attn_scores, interpolated_hidden, adaptive_factor,
                attended, attn_weights, imputed_missed_x, x_imputed)
                
     @torch.jit.script_method
    def SelectiveSSM(self, x: torch.Tensor, delta_time: torch.Tensor, prev_latent: torch.Tensor):
        """
        x_t: - contains NaNs
        delta_time: - time since last step
        prev_latent: - prev_latent
        """
        # 1. Create Observation Mask
        mask = ~torch.isnan(x)  # True where data exists
        mask_float = mask.float()
        # 2. Neutralize NaNs for the Linear layer
        x_filled = torch.nan_to_num(x, nan=0.0)
        # 3. Inform the projection about what is missing# Concatenate values and mask: [B, F*2]
        selective_input = torch.cat([x_filled, mask_float], dim=-1)
        # 4. Generate SSM parameters (Delta, B, C)
        projected= torch.einsum("bij,jik->bjk", selective_input.unsqueeze(1), self.x_proj)
        projected_back = torch.tanh(torch.einsum("bth,tfh->bfh", projected, self.x_proj_back))
        dt_learned, B_t, C_t = torch.split(projected_back, self.hidden_size, dim=-1)
        # Discretization parameters
        dt = F.softplus(dt_learned + delta_time)
        lam = F.softplus(self.log_lambda)
        dA = torch.exp(-lam * dt)
        dB = dt * B_t
        # 5. MASKED STATE UPDATE (The Critical Part)
        # We only add (dB * x) if the mask is 1.# If the mask is 0, the state only decays via dA.
        update_term = dB * x_filled.unsqueeze(-1)

        # We use the mask to gate the update#
        # If mask is 0 for a feature, that specific feature dimension doesn't update the latent
        temporal_context = torch.einsum("bth,hk->btk", dA, self.time_weight) + self.time_bias
        new_latent = temporal_context + (mask_float.mean(dim=-1, keepdim=True).unsqueeze(1) * update_term)

        #new_latent = (dA + prev_latent) + (mask_float.mean(dim=-1, keepdim=True).unsqueeze(1) * update_term)
        # 6. Output Selection
        y_t = new_latent * torch.sigmoid(C_t)
        return dA, lam, new_latent, y_t

    @torch.jit.script_method
    def tsmta(self, decay_weight, c_tilda, time, h_tilda):
        # Setp 1 Semantic projection & Normalize to get importance weights
        s_t = torch.tanh(torch.einsum("bij,jik->bjk", decay_weight, self.U_s) + \
                         torch.einsum("bij,ijk->bik", c_tilda, self.W_s) + self.b_s)
        alpha_s_t = F.softmax(s_t, dim=-1)
        # Step 2 Decay-Aware based Temporal decay factor (decay_weight derived)
        phi_t = alpha_s_t * decay_weight.permute(0, 2, 1)
        # Setp 3 Semantic Trace Update based on the elasped between events and previous memory cell
        gamma_input = torch.cat([decay_weight.permute(0, 2, 1), time, phi_t], dim=-1)
        gamma_t = torch.sigmoid(torch.einsum("bij,jik->bjk", gamma_input.permute(0, 2, 1), self.U_gamma) +
                                torch.einsum("bij,ijk->bik", c_tilda, self.W_gamma))
        epsilon = torch.randn_like(gamma_t) * 0.05  # small noise
        gamma_t = torch.clamp(gamma_t + epsilon, 0, 1)
        # Setp 4 Drift Gate based on Semantic Trace, Decay-Aware & previous memory h_tilda
        trace_t = gamma_t * phi_t + (1 - gamma_t) * h_tilda
        # Setp 5 Memory combination (trace vs. old cell)
        alpha_input = torch.cat([trace_t, c_tilda], dim=-1)
        alpha_t = torch.sigmoid(torch.einsum("bij,jik->bjk", alpha_input.permute(0, 2, 1), self.U_gamma) +
                                torch.einsum("bij,ijk->bik", c_tilda, self.W_gamma))
        # Setp 6  Aggregated Cell Memory
        c_tilde = alpha_t * trace_t + (1 - alpha_t) * c_tilda
        return c_tilde, alpha_s_t

    @torch.jit.script_method
    def modulate_hidden(self, h: torch.Tensor, freq_enc: torch.Tensor, time_enc: torch.Tensor):
        # Expand freq to match h shape: [B, T, H]
        gate_input = torch.cat([h, freq_enc, time_enc], dim=-1)
        # print("gate_input", gate_input.shape)
        gate = torch.sigmoid(torch.einsum("bij,ijk->bik", gate_input, self.W_in_imp))  # [B, T, H]
        return gate * h + (1 - gate) * freq_enc  # Interpolation

    @torch.jit.script_method
    def time_decay_encode(self, t: torch.Tensor):
        t_log = torch.div(self.c1_const, torch.log(t + self.c2_const))
        time_features = torch.stack([t_log, torch.sin(t_log),
                                     torch.cos(t_log)],
                                    dim=-1)
        # Project to hidden size
        time_proj = torch.sigmoid(torch.einsum("bij,ijk->bik", time_features,  self.ones_const))
        return time_proj

    @torch.jit.script_method
    def freq_decaying(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_log = torch.exp(-self.factor_impu * freq_dict)  # [B, F]
        freq_norm = (freq_log - freq_log.mean()) / (freq_log.std() + 1e-6)
        freq_features = torch.stack([freq_norm, torch.sin(freq_norm),
                                     torch.cos(freq_norm)], dim=-1)  # [B, F, 4] .unsqueeze(-1)
        weights = torch.sigmoid(torch.einsum("bij,ijk->bik", freq_features, self.Dw) + \
                                torch.einsum("bij,ijk->bik", ht, self.W_d) + self.b_d)
        return weights

    @torch.jit.script_method
    def freq_encode(self, freq_dict: torch.Tensor):
        # normalize and apply continuous basis functions (e.g., Fourier + polynomial)
        freq_log = torch.exp(-self.factor_impu * freq_dict)  # [B, F]
        freq_norm = (freq_log - freq_log.mean()) / (freq_log.std() + 1e-6)
        freq_features = torch.stack([freq_norm, torch.sin(freq_norm),
                                     torch.cos(freq_norm)],dim=-1)  # [B, F, 4] .unsqueeze(-1)
        # Project to hidden size
        freq_proj = torch.sigmoid(torch.einsum("bij,ijk->bik", freq_features, self.Dw))
        return freq_proj  # [B, F, H]

class GatedTemporalAttentionOutput(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_dim):
        super().__init__()
        self.proj = nn.Linear(seq_len, 4 * hidden_size)  # temporal to hidden
        self.norm = nn.LayerNorm([input_size, 4 * hidden_size])

        # Gated attention to summarize across time
        self.attn = nn.Sequential(
            nn.Linear(4 * hidden_size, 1),
            nn.Sigmoid()  # soft gate across time
        )

        self.output = nn.Linear(4 * hidden_size, output_dim)

    def forward(self, x):  # x: [B, I, T] = [B, input_size, seq_len]
        x = self.proj(x)  # [B, I, 4H]
        x = self.norm(x)  # [B, I, 4H]

        attn = self.attn(x)  # [B, I, 1]
        pooled = (x * attn).sum(dim=1)  # [B, 4H]

        return self.output(pooled)  # [B, output_dim]

class ContextConditioned(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_dim, num_freqs=16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Pre-normalization for stability
        self.pre_norm = nn.LayerNorm(input_size)

        # === Stable Temporal Mixing ===
        # Fourier mixing (learned freq embeddings instead of conv)
        self.freqs = nn.Parameter(torch.randn(num_freqs))
        self.freq_proj = nn.Linear(num_freqs * 2, input_size)

        # Channel mixing with residual scaling
        self.channel_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # stabilize residual

        # Diffusion conditioning
        self.diff_proj = nn.Linear(input_size, hidden_size)

        # Semantic salience gating (with dropout to avoid gate collapse)
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, sigma=None):
        # x: [B, I, T]
        B, I, T = x.shape
        # Feature normalization
        x = x.permute(0, 2, 1)   # [B, T, I]
        x = self.pre_norm(x)
        # === Fourier temporal embedding ===
        t = torch.linspace(0, 1, T, device=x.device).unsqueeze(-1)  # [T,1]
        freqs = self.freqs[None, None, :] * t  # [1,T,F]
        fourier_basis = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [1,T,2F]
        temporal_mix = self.freq_proj(fourier_basis).expand(B, -1, -1)  # [B,T,I]
        # Channel projection + residual
        x_proj = self.channel_proj(x + temporal_mix)  # [B,T,H]

        if sigma is not None:
            cond = self.diff_proj(torch.tanh(sigma))  # [B,T,H]
            x_proj = x_proj + cond

        x_proj = x_proj + self.res_scale * x_proj  # stabilize residual
        # Gated pooling
        # Softmax semantic gate
        attn = F.softmax(self.semantic_gate(x_proj).squeeze(-1), dim=-1)  # [B, T]
        pooled = torch.sum(x_proj * attn, dim=1)  # [B, H]

        return self.out_proj(pooled)

class GTACM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_steps,
                 num_layers, output_dim, device, batch_first=True,
                 bidirectional=True):
        super(GTACM, self).__init__()
        # hidden dimensions
        self.device = device
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        self.initializer_range = 0.02
        self.num_layers=num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # Gated Temporal Attractor Cell
        # Create multi-layer bidirectional ATRUCell stacks
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            # Keep input_size constant at input_dim for all layers
            # This allows freq/last_values to remain consistent
            self.layers.append(nn.ModuleList([
                ATRUCell(self.input_size, self.hidden_size, self.seq_len),
                ATRUCell(self.input_size, self.hidden_size, self.seq_len) if bidirectional else None
            ]))

        # Projection layers to convert hidden states to input_dim for the next layer
        if self.num_layers > 1:
            self.layer_projections = nn.ModuleList()
            for layer in range(self.num_layers - 1):
                # Project from (hidden_dim * num_directions) -> input_dim
                # After averaging over features, we go from [seq*batch, hidden_dim*num_directions]
                # to [seq*batch, input_dim]
                self.layer_projections.append(
                    nn.Linear(self.hidden_size * self.num_directions, self.input_size)
                )
        self.dropout_layer = nn.Dropout(0.2)
        # DiffusionEmbedding
        self.diffusion_embedding = AdaptiveNoiseEmbedding(self.seq_len, self.input_size, self.device)
        self.F_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                 size=(self.input_size, self.hidden_size * 2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                     size=(self.input_size, 1)))
        self.F_beta = nn.Linear(self.seq_len, self.hidden_size)
        self.layer_norm1 = nn.LayerNorm([self.input_size, self.seq_len])
        self.layer_norm = nn.LayerNorm([self.input_size, self.hidden_size])
        self.Phi = nn.Linear(self.hidden_size, self.seq_len)
        self.output_phi = nn.Linear(self.seq_len, self.output_dim)
        self.out_norm = nn.LayerNorm([self.input_size, self.seq_len])
        self.PhiOut = nn.Linear(self.seq_len, self.output_dim)
        self.output_layer = ContextConditioned(seq_len=self.seq_len,
                                               input_size=self.input_size,
                                               hidden_size=self.hidden_size,
                                               output_dim=self.output_dim)
    def forward(self, inputs, times, last_values, freqs):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]

        seq_len = inputs.size()[0]
        final_h = torch.jit.annotate(List[Tensor], [])

        # Initialize output variables before the loop

        hidden_his = None
        imputed_inputs = None
        attns_t_facit_weights = None
        weights_decay = None
        attended_crossW_f = None
        attns_tsma_weights = None
        interpolated_hidden = None
        adaptive_factor = None
        attns_cross_f=None
        freqs_facit_decay = None
        freqs_decay_slope = None
        # Process through each layer
        layer_inputs = inputs  # [seq_len, batch, input_dim]
        for layer_idx, (f_cell, b_cell) in enumerate(self.layers):
            prev_hidden = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
            prev_cell = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
            hidden_his = torch.jit.annotate(List[Tensor], [])
            imputed_inputs = torch.jit.annotate(List[Tensor], [])
            attns_t_facit_weights = torch.jit.annotate(List[Tensor], [])
            weights_decay = torch.jit.annotate(List[Tensor], [])
            attended_crossW_f = torch.jit.annotate(List[Tensor], [])
            attns_tsma_weights = torch.jit.annotate(List[Tensor], [])
            attns_cross_f = torch.jit.annotate(List[Tensor], [])
            freqs_facit_decay = torch.jit.annotate(List[Tensor], [])
            freqs_decay_slope = torch.jit.annotate(List[Tensor], [])
            interpolated_hidden = torch.jit.annotate(List[Tensor], [])
            adaptive_factor = torch.jit.annotate(List[Tensor], [])

            # Forward pass
            for i in range(seq_len):
                (prev_hidden, prev_cell, attn_t_facit,
                 fre_decay, attended_cross_f, alpha_tsma_f,
                 attn_cross_f, freq_facit,
                 decay_slope,interpolated, adaptive, imputed_x) = f_cell(
                    prev_hidden, prev_cell,
                    layer_inputs[i], times[i],
                    last_values[i], freqs[i]
                )
                hidden_his += [prev_hidden]
                attns_t_facit_weights += [attn_t_facit]
                weights_decay += [fre_decay]
                attended_crossW_f += [attended_cross_f]
                attns_tsma_weights += [alpha_tsma_f]
                attns_cross_f += [attn_cross_f]
                freqs_facit_decay += [freq_facit]
                freqs_decay_slope += [decay_slope]
                imputed_inputs += [imputed_x]
                interpolated_hidden += [interpolated]
                adaptive_factor += [adaptive]

            imputed_inputs = torch.stack(imputed_inputs)
            hidden_his = torch.stack(hidden_his)
            attns_t_facit_weights = torch.stack(attns_t_facit_weights)
            weights_decay = torch.stack(weights_decay)
            attended_crossW_f = torch.stack(attended_crossW_f)
            attns_tsma_weights = torch.stack(attns_tsma_weights)
            attns_cross_f = torch.stack(attns_cross_f)
            freqs_facit_decay = torch.stack(freqs_facit_decay)
            freqs_decay_slope = torch.stack(freqs_decay_slope)
            interpolated_hidden = torch.stack(interpolated_hidden)
            adaptive_factor = torch.stack(adaptive_factor)
            # Bidirectional backward pass
            if self.bidirectional:
                second_hidden = torch.zeros((batch_size, self.input_size,
                                             self.hidden_size), device=device)
                second_cell = torch.zeros((batch_size, self.input_size,
                                           self.hidden_size), device=device)
                sc_inputs = torch.flip(layer_inputs, [0])
                sc_times = torch.flip(times, [0])
                imputed_inputs_sec = torch.jit.annotate(List[Tensor], [])
                second_hidden_his = torch.jit.annotate(List[Tensor], [])
                attns_t_facit_weights_b = torch.jit.annotate(List[Tensor], [])
                weights_decay_b = torch.jit.annotate(List[Tensor], [])
                attended_crossW_f_b = torch.jit.annotate(List[Tensor], [])
                attns_tsma_weights_b = torch.jit.annotate(List[Tensor], [])
                attns_cross_f_b = torch.jit.annotate(List[Tensor], [])
                freqs_facit_decay_b = torch.jit.annotate(List[Tensor], [])
                freqs_decay_slope_b = torch.jit.annotate(List[Tensor], [])
                interpolated_hidden_b = torch.jit.annotate(List[Tensor], [])
                adaptive_factor_b = torch.jit.annotate(List[Tensor], [])

                for i in range(seq_len):
                    time = sc_times[i]
                    (second_hidden, second_cell, attn_t_facit_b,
                     fre_decay_b,  attended_cross_f_b, alpha_tsma_f_b,
                     attn_cross_f_b, freq_facit_b,
                     decay_slope_b, interpolated_b,adaptive_b, imputed_x_b) = b_cell(
                        second_hidden, second_cell,
                        sc_inputs[i], time,
                        last_values[i], freqs[i]
                    )
                    second_hidden_his += [second_hidden]
                    imputed_inputs_sec += [imputed_x_b]
                    attns_t_facit_weights_b += [attn_t_facit_b]
                    weights_decay_b += [fre_decay_b]
                    attended_crossW_f_b += [attended_cross_f_b]
                    attns_tsma_weights_b += [alpha_tsma_f_b]
                    attns_cross_f_b += [attn_cross_f_b]
                    freqs_facit_decay_b += [freq_facit_b]
                    freqs_decay_slope_b += [decay_slope_b]
                    interpolated_hidden_b += [interpolated_b]
                    adaptive_factor_b += [adaptive_b]

                imputed_inputs_sec = torch.stack(imputed_inputs_sec)
                second_hidden_his = torch.stack(second_hidden_his)
                attns_t_facit_weights_b = torch.stack(attns_t_facit_weights_b)
                weights_decay_b = torch.stack(weights_decay_b)
                attended_crossW_f_b = torch.stack(attended_crossW_f_b)
                attns_tsma_weights_b = torch.stack(attns_tsma_weights_b)
                attns_cross_f_b = torch.stack(attns_cross_f_b)
                freqs_facit_decay_b = torch.stack(freqs_facit_decay_b)
                freqs_decay_slope_b = torch.stack(freqs_decay_slope_b)
                interpolated_hidden_b = torch.stack(interpolated_hidden_b)
                adaptive_factor_b = torch.stack(adaptive_factor_b)
                # Flip backward results back to forward order
                imputed_inputs_sec = torch.flip(imputed_inputs_sec, [0])
                second_hidden_his = torch.flip(second_hidden_his, [0])
                attns_t_facit_weights_b = torch.flip(attns_t_facit_weights_b, [0])
                weights_decay_b = torch.flip(weights_decay_b, [0])
                attended_crossW_f_b = torch.flip(attended_crossW_f_b, [0])
                attns_tsma_weights_b = torch.flip(attns_tsma_weights_b, [0])
                attns_cross_f_b = torch.flip(attns_cross_f_b, [0])
                freqs_facit_decay_b = torch.flip(freqs_facit_decay_b, [0])
                freqs_decay_slope_b = torch.flip(freqs_decay_slope_b, [0])
                interpolated_hidden_b = torch.flip(interpolated_hidden_b, [0])
                adaptive_factor_b = torch.flip(adaptive_factor_b, [0])

                # Concatenate forward and backward hidden states
                hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
                imputed_inputs = torch.cat((imputed_inputs, imputed_inputs_sec), dim=2)
                attns_t_facit_weights = torch.cat((attns_t_facit_weights, attns_t_facit_weights_b), dim=-1)
                weights_decay = torch.cat((weights_decay, weights_decay_b), dim=-1)
                attended_crossW_f = torch.cat((attended_crossW_f, attended_crossW_f_b), dim=2)
                interpolated_hidden = torch.cat((interpolated_hidden, interpolated_hidden_b), dim=2)
                adaptive_factor = torch.cat((adaptive_factor, adaptive_factor_b), dim=2)
                attns_tsma_weights = torch.cat((attns_tsma_weights, attns_tsma_weights_b), dim=-1)
                attns_cross_f = torch.cat((attns_cross_f, attns_cross_f_b), dim=0)
                freqs_facit_decay = torch.cat((freqs_facit_decay, freqs_facit_decay_b), dim=0)
                freqs_decay_slope = torch.cat((freqs_decay_slope.unsqueeze(-1),
                                               freqs_decay_slope_b.unsqueeze(-1)), dim=-1)

            final_h.append(hidden_his)
            # Apply dropout except for the last layer
            if self.dropout_layer is not None and layer_idx < len(self.layers) - 1:
                layer_inputs = self.dropout_layer(layer_inputs)

            # Prepare output for the next layer
            if layer_idx < len(self.layers) - 1:
                seq_len_out = hidden_his.size(0)
                batch_out = hidden_his.size(1)
                features = hidden_his.size(2)
                hidden_combined = hidden_his.size(3)
                # Pool across features: average the hidden states for each feature
                pooled_hidden = hidden_his.mean(dim=2)  # Average over features
                # Reshape: [seq_len, batch, hidden_dim*num_directions]
                #       -> [seq_len*batch, hidden_dim*num_directions]
                hidden_reshaped = pooled_hidden.reshape(seq_len_out * batch_out, hidden_combined)
                # Project to input_dim size
                projected = self.layer_projections[layer_idx](hidden_reshaped)
                # Reshape back: [seq_len, batch, input_dim]
                layer_inputs = projected.reshape(seq_len_out, batch_out, self.input_size)
        # Ensure variables are defined before final processing
        if hidden_his is None or imputed_inputs is None or weights_decay is None or \
                attns_t_facit_weights is None or attended_crossW_f  is None or \
                attns_tsma_weights is None or attns_cross_f is None or \
                freqs_facit_decay is None or freqs_decay_slope is None:
            raise RuntimeError("No layers were processed in forward pass")

        if self.batch_first:
            hidden_his = final_h[-1].permute(1, 0, 2, 3)
            imputed_inputs = imputed_inputs.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 3, 2)
            attns_cross_f = attns_cross_f.permute(1, 0, 2, 3, 4)
            attended_crossW_f = attended_crossW_f.permute(1, 0, 2, 3)
            interpolated_hidden=interpolated_hidden.permute(1, 0, 2, 3)
            adaptive_factor = adaptive_factor.permute(1, 0, 2, 3)
            attns_t_facit_weights = attns_t_facit_weights.permute(1, 0,3, 2)
            attns_tsma_weights = attns_tsma_weights.permute(1, 0, 2, 3)
            freqs_facit_decay = freqs_facit_decay.permute(1, 0, 2)
            freqs_decay_slope = freqs_decay_slope.permute(2, 0, 1)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha) + self.F_alpha_n_b)
        alphas = alphas.reshape(alphas.size(0), alphas.size(2), alphas.size(1) * alphas.size(-1))
        x = self.layer_norm1(alphas)  # [B, D, L]
        x = self.F_beta(x)  # [B, D, 4H]
        x = self.Phi(self.layer_norm(x))
        out = self.output_layer(x)
        return (out, weights_decay,attns_cross_f, attns_t_facit_weights,
                attns_tsma_weights,attended_crossW_f, freqs_facit_decay,
                freqs_decay_slope, interpolated_hidden, adaptive_factor,
                hidden_his, imputed_inputs)

class GTACMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, diff_step,num_layers, output_dim, device):
        super(GTACMNetwork, self).__init__()
        # hidden dimensions
        self.device = device
        self.seq_len = seq_len
        self.input_size = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_steps = diff_step
        # Gated Temporal Attractor Cell
        self.gtacm = GTACM(self.input_size, self.hidden_size, self.seq_len,
                           self.num_steps, self.num_layers, self.output_dim,
                           self.device)

    def forward(self, historic_features, timestamp, last_features, features_freqs):
        # Temporal features embedding
        (out, freq_w , att_fw, attns_fw,
         attn_weights, attended_fw,
         freqs_facit_decay, freqs_decay_slope,interpolated_hidden,
         adaptive_factor, hidden_his, imputed_inputs) = self.gtacm(historic_features, timestamp,
                                      last_features, features_freqs,
                                      )

        return (out,freq_w, att_fw, attns_fw, attn_weights,
                attended_fw, freqs_facit_decay,
                freqs_decay_slope, interpolated_hidden,
                adaptive_factor,hidden_his, imputed_inputs,
                imputed_inputs.mean(axis=2))