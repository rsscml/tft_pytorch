"""
PyTorch implementations of the Temporal Fusion Transformer (TFT) family.

Classes
-------
TemporalFusionTransformer
    Full encoder-decoder TFT for multi-horizon, multi-quantile forecasting.
TFTEncoderOnly
    Encoder-only variant for tasks with no future inputs (regression /
    classification from historical windows).

All TFT building-block layers are also exported for custom architectures:
    apply_time_distributed, scaled_dot_product_attention,
    TFTMultiHeadAttention, TFTLinearLayer, TFTApplyMLP,
    TFTApplyGatingLayer, TFTAddAndNormLayer, TFTGRNLayer,
    VariableSelectionStatic, VariableSelectionTemporal,
    StaticContexts, LSTMLayer, StaticEnrichmentLayer,
    AttentionLayer, AttentionStack, FinalGatingLayer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def apply_time_distributed(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply *module* independently to every time-step (imitates Keras TimeDistributed)."""
    if x.ndimension() <= 2:
        return module(x)
    b, t, *others = x.size()
    y = module(x.reshape(b * t, *others))
    if y.ndimension() == 2:
        return y.reshape(b, t, y.size(-1))
    return y.reshape(b, t, *y.shape[1:])


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """Return [B, 1, T] mask where 1 = position to block."""
    return (seq < 0).float().unsqueeze(1)


def causal_mask(size: int) -> torch.Tensor:
    """Return [size, size] upper-triangular mask (1 = block future)."""
    return torch.ones(size, size).triu(diagonal=1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard scaled dot-product attention."""
    dk = k.size(-1)
    logits = torch.matmul(q, k.transpose(-1, -2)) / (dk ** 0.5)
    if causal_mask is not None:
        logits = logits + causal_mask * -1e9
    if padding_mask is not None:
        logits = logits + padding_mask * -1e9
    weights = F.softmax(logits, dim=-1)
    return torch.matmul(weights, v), weights


class TFTMultiHeadAttention(nn.Module):
    """
    Multi-head attention following the original TFT formulation where
    the value projection is *shared* across all heads.
    """

    def __init__(self, n_head: int, d_model: int, device: str, dropout_rate: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.device = device

        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(device)
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(device)
        vs_layer = nn.Linear(d_model, self.d_k, bias=False).to(device)
        self.vs_layers = nn.ModuleList([vs_layer] * n_head)

        self.dropout = nn.Dropout(dropout_rate)
        self.w_o = nn.Linear(d_model, d_model, bias=False).to(device)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads, attns = [], []
        for i in range(self.n_head):
            h, a = scaled_dot_product_attention(
                self.qs_layers[i](q),
                self.ks_layers[i](k),
                self.vs_layers[i](v),
                causal_mask, padding_mask,
            )
            heads.append(self.dropout(h))
            attns.append(a)
        out = self.w_o(torch.cat(heads, dim=-1))
        return self.dropout(out), torch.stack(attns, dim=0)


# ---------------------------------------------------------------------------
# Core TFT building blocks
# ---------------------------------------------------------------------------

def get_activation_fn(name: Optional[str]):
    if name is None:
        return None
    name = name.lower()
    if name == 'elu':
        return F.elu
    if name == 'tanh':
        return torch.tanh
    if name == 'sigmoid':
        return torch.sigmoid
    if name == 'softmax':
        return F.softmax
    raise ValueError(f"Unsupported activation: {name}")


class TFTLinearLayer(nn.Module):
    """Dense layer with lazy input-dimension inference and optional time-distribution."""

    def __init__(self, hidden_layer_size: int, device: str, activation=None,
                 use_time_distributed: bool = False, use_bias: bool = True):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.use_time_distributed = use_time_distributed
        self.activation_fn = get_activation_fn(activation)
        self.layer = nn.Linear(1, 1, bias=use_bias).to(device)  # rebuilt lazily
        self._built = False

    def _build(self, in_features: int):
        self.layer = nn.Linear(in_features, self.hidden_layer_size, bias=self.layer.bias is not None).to(self.device)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
        out = apply_time_distributed(self.layer, x) if self.use_time_distributed else self.layer(x)
        return self.activation_fn(out) if self.activation_fn else out


class TFTApplyGatingLayer(nn.Module):
    """GLU-style gating: output = activation(Wx) ⊙ σ(Vx)."""

    def __init__(self, hidden_layer_size: int, device: str, dropout_rate: float = 0.0,
                 use_time_distributed: bool = True, activation=None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.use_time_distributed = use_time_distributed
        self.activation_fn = get_activation_fn(activation)
        self.activation_fc = nn.Linear(1, 1).to(device)
        self.gated_fc = nn.Linear(1, 1).to(device)
        self._built = False

    def _build(self, in_features: int):
        self.activation_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
        self.gated_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
        self._built = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._built:
            self._build(x.shape[-1])
        x = self.dropout(x)
        if self.use_time_distributed:
            a = apply_time_distributed(self.activation_fc, x)
            g = apply_time_distributed(self.gated_fc, x)
        else:
            a = self.activation_fc(x)
            g = self.gated_fc(x)
        if self.activation_fn:
            a = self.activation_fn(a)
        g = torch.sigmoid(g)
        return a * g, g


class TFTAddAndNormLayer(nn.Module):
    """Residual add + layer normalisation."""

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.layer_norm: Optional[nn.LayerNorm] = None

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        skip, gate = inputs
        out = skip + gate
        if self.layer_norm is None:
            self.layer_norm = nn.LayerNorm(out.shape[-1]).to(self.device)
        return self.layer_norm(out)


class TFTApplyMLP(nn.Module):
    """Two-layer MLP with configurable activations."""

    def __init__(self, hidden_size: int, output_size: int,
                 output_activation=None, hidden_activation: str = 'tanh',
                 use_time_distributed: bool = False):
        super().__init__()
        self.use_time_distributed = use_time_distributed
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_act = torch.tanh if hidden_activation == 'tanh' else F.elu
        self.out = nn.Linear(hidden_size, output_size)
        self.out_act = (None if output_activation is None
                        else torch.sigmoid if output_activation == 'sigmoid'
                        else (lambda x: F.softmax(x, dim=-1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_time_distributed:
            x = self.hidden_act(apply_time_distributed(self.hidden, x))
            x = apply_time_distributed(self.out, x)
        else:
            x = self.hidden_act(self.hidden(x))
            x = self.out(x)
        return self.out_act(x) if self.out_act else x


class TFTGRNLayer(nn.Module):
    """Gated Residual Network (GRN) as described in the TFT paper."""

    def __init__(self, device: str, hidden_layer_size: int, output_size: Optional[int] = None,
                 dropout_rate: float = 0.0, use_time_distributed: bool = True,
                 additional_context: bool = False, return_gate: bool = False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size or hidden_layer_size
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        self.return_gate = return_gate
        self.device = device

        self.skip_layer = nn.Linear(hidden_layer_size, self.output_size)
        self.hidden_1 = TFTLinearLayer(hidden_layer_size, device=device, use_time_distributed=use_time_distributed)
        self.hidden_2 = TFTLinearLayer(hidden_layer_size, device=device, use_time_distributed=use_time_distributed)

        if additional_context:
            self.context_layer = TFTLinearLayer(hidden_layer_size, device=device,
                                                use_time_distributed=use_time_distributed, use_bias=False)
        else:
            self.context_layer = None

        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.output_size, dropout_rate=dropout_rate,
                                        use_time_distributed=use_time_distributed, device=device)
        self.add_norm = TFTAddAndNormLayer(device=device)

    def forward(self, inputs):
        if self.additional_context:
            x, c = inputs
        else:
            x = inputs

        skip_out = apply_time_distributed(self.skip_layer, x) if self.use_time_distributed else self.skip_layer(x)

        h = self.hidden_1(x)
        if self.context_layer is not None:
            if h.ndimension() == 3 and c.ndimension() == 2:
                c = c.unsqueeze(1)
            h = h + self.context_layer(c)

        h = F.elu(h)
        h = self.hidden_2(h)
        gating_out, gate = self.gate(h)
        out = self.add_norm([skip_out, gating_out])

        return (out, gate) if self.return_gate else out


# ---------------------------------------------------------------------------
# Variable selection, contexts, LSTM, enrichment
# ---------------------------------------------------------------------------

class VariableSelectionStatic(nn.Module):
    """Variable selection network for static (non-temporal) inputs."""

    def __init__(self, hidden_layer_size: int, dropout_rate: float, device: str):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.grn_flat: Optional[TFTGRNLayer] = None
        self.grn_vars = nn.ModuleList()

    def _build(self, num_vars: int):
        self.grn_flat = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size, output_size=num_vars,
                                    dropout_rate=self.dropout_rate, use_time_distributed=False,
                                    device=self.device).to(self.device)
        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size, output_size=None,
                            dropout_rate=self.dropout_rate, use_time_distributed=False,
                            device=self.device).to(self.device)
            )

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.grn_flat is None:
            self._build(len(inputs))
        flat = torch.cat(inputs, dim=1)
        weights = F.softmax(self.grn_flat(flat), dim=-1).unsqueeze(-1)
        transformed = torch.stack([self.grn_vars[i](inputs[i]) for i in range(len(inputs))], dim=1)
        return (transformed * weights).sum(dim=1), weights.squeeze(-1)


class StaticContexts(nn.Module):
    """Produce four context vectors from the static representation."""

    def __init__(self, hidden_layer_size: int, dropout_rate: float, device: str):
        super().__init__()
        kw = dict(hidden_layer_size=hidden_layer_size, output_size=None, dropout_rate=dropout_rate,
                  use_time_distributed=False, device=device)
        self.stat_vec_layer = TFTGRNLayer(**kw)
        self.enrich_vec_layer = TFTGRNLayer(**kw)
        self.h_vec_layer = TFTGRNLayer(**kw)
        self.c_vec_layer = TFTGRNLayer(**kw)

    def forward(self, inputs: torch.Tensor):
        return (self.stat_vec_layer(inputs), self.enrich_vec_layer(inputs),
                self.h_vec_layer(inputs), self.c_vec_layer(inputs))


class VariableSelectionTemporal(nn.Module):
    """Variable selection network for temporal (sequential) inputs."""

    def __init__(self, hidden_layer_size: int, static_context: bool, dropout_rate: float, device: str):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.static_context = static_context
        self.dropout_rate = dropout_rate
        self.device = device
        self.grn_flat: Optional[TFTGRNLayer] = None
        self.grn_vars = nn.ModuleList()

    def _build(self, num_vars: int):
        self.grn_flat = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size, output_size=num_vars,
                                    dropout_rate=self.dropout_rate, use_time_distributed=True,
                                    additional_context=self.static_context, device=self.device).to(self.device)
        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size, output_size=None,
                            dropout_rate=self.dropout_rate, use_time_distributed=True,
                            device=self.device).to(self.device)
            )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.static_context:
            inputs, context = x
        else:
            inputs = x
        if self.grn_flat is None:
            self._build(len(inputs))

        flat = torch.cat(inputs, dim=-1)
        mlp_out = self.grn_flat((flat, context.unsqueeze(1))) if self.static_context else self.grn_flat(flat)
        weights = F.softmax(mlp_out, dim=-1).unsqueeze(-1)
        transformed = torch.stack([self.grn_vars[i](inputs[i]) for i in range(len(inputs))], dim=2)
        return (weights * transformed).sum(dim=2), weights.squeeze(-1)


class LSTMLayer(nn.Module):
    """Encoder-decoder LSTM pair with gating and residual connection."""

    def __init__(self, hidden_layer_size: int, device: str, rnn_layers: int, dropout_rate: float):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.device = device
        self.tft_encoder = nn.LSTM(hidden_layer_size, hidden_layer_size, rnn_layers, batch_first=True)
        self.tft_decoder = nn.LSTM(hidden_layer_size, hidden_layer_size, rnn_layers, batch_first=True)
        self.gate = TFTApplyGatingLayer(hidden_layer_size, dropout_rate=dropout_rate,
                                        use_time_distributed=True, device=device)
        self.add_norm = TFTAddAndNormLayer(device=device)

    def forward(self, inputs):
        enc_in, dec_in, (h0, c0) = inputs
        h0 = h0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)
        enc_out, (eh, ec) = self.tft_encoder(enc_in, (h0, c0))
        dec_out, _ = self.tft_decoder(dec_in, (eh, ec))
        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        lstm_in = torch.cat([enc_in, dec_in], dim=1)
        gated, _ = self.gate(lstm_out)
        return self.add_norm([gated, lstm_in])


class StaticEnrichmentLayer(nn.Module):
    """Enrich temporal features with static context via a GRN."""

    def __init__(self, hidden_layer_size: int, context: bool, dropout_rate: float, device: str):
        super().__init__()
        self.context = context
        self.grn = TFTGRNLayer(hidden_layer_size=hidden_layer_size, output_size=None,
                               dropout_rate=dropout_rate, use_time_distributed=True,
                               additional_context=context, device=device)

    def forward(self, inputs):
        if self.context:
            x, c = inputs
            return self.grn((x, c.unsqueeze(1)))
        return self.grn(inputs)


# ---------------------------------------------------------------------------
# Attention stack
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_layer_size: int, device: str, n_head: int, dropout_rate: float):
        super().__init__()
        self.mha = TFTMultiHeadAttention(n_head, hidden_layer_size, dropout_rate=dropout_rate, device=device)
        self.gate = TFTApplyGatingLayer(hidden_layer_size, dropout_rate=dropout_rate,
                                        use_time_distributed=True, device=device)
        self.add_norm = TFTAddAndNormLayer(device=device)

    def forward(self, x, attn_mask, padding_mask):
        out, _ = self.mha(x, x, x, attn_mask, padding_mask)
        out, _ = self.gate(out)
        return self.add_norm([out, x])


class AttentionStack(nn.Module):
    def __init__(self, num_layers: int, hidden_layer_size: int, n_head: int,
                 dropout_rate: float, device: str):
        super().__init__()
        self.layers_ = [AttentionLayer(hidden_layer_size, device, n_head, dropout_rate)
                        for _ in range(num_layers)]
        self.grn_final = TFTGRNLayer(hidden_layer_size=hidden_layer_size, output_size=None,
                                     dropout_rate=dropout_rate, use_time_distributed=True,
                                     device=device)

    def forward(self, x, attn_mask, padding_mask):
        for layer in self.layers_:
            x = layer(x, attn_mask, padding_mask)
        return self.grn_final(x)


class FinalGatingLayer(nn.Module):
    def __init__(self, hidden_layer_size: int, device: str, dropout_rate: float):
        super().__init__()
        self.gate = TFTApplyGatingLayer(hidden_layer_size, dropout_rate=dropout_rate,
                                        use_time_distributed=True, device=device)
        self.add_norm = TFTAddAndNormLayer(device=device)

    def forward(self, inputs):
        attn_out, temporal = inputs
        gated, _ = self.gate(attn_out)
        return self.add_norm([gated, temporal])


# ---------------------------------------------------------------------------
# Full TFT model
# ---------------------------------------------------------------------------

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon, multi-quantile forecasting.

    Architecture (per the original paper):
    1. Categorical embedding + continuous projection for all inputs
    2. Variable selection networks (static, historical, future)
    3. Static context encoding (4 context vectors)
    4. LSTM encoder-decoder
    5. Static enrichment of temporal features
    6. Multi-head self-attention stack (with causal mask)
    7. Final gating + position-wise GRN
    8. Output projection for each quantile

    Parameters
    ----------
    hidden_layer_size : int
    num_attention_heads : int
    num_lstm_layers : int
    num_attention_layers : int
    dropout_rate : float
    num_static_categorical : int
    num_static_continuous : int
    num_historical_categorical : int
    num_historical_continuous : int
    num_future_categorical : int
    num_future_continuous : int
    categorical_embedding_dims : dict, optional
        ``{var_name: (vocab_size, embed_dim)}``
    historical_steps : int
    prediction_steps : int
    num_outputs : int
        Number of quantiles to predict (e.g. 3 for [0.1, 0.5, 0.9]).
    device : str
    """

    def __init__(
        self,
        hidden_layer_size: int = 160,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 1,
        num_attention_layers: int = 1,
        dropout_rate: float = 0.1,
        num_static_categorical: int = 0,
        num_static_continuous: int = 0,
        num_historical_categorical: int = 0,
        num_historical_continuous: int = 0,
        num_future_categorical: int = 0,
        num_future_continuous: int = 0,
        categorical_embedding_dims: Optional[Dict] = None,
        historical_steps: int = 168,
        prediction_steps: int = 24,
        num_outputs: int = 3,
        device: str = 'cpu',
    ):
        super().__init__()

        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.num_outputs = num_outputs

        # Embeddings
        self.categorical_embeddings = nn.ModuleDict()
        if categorical_embedding_dims:
            for name, (vocab, dim) in categorical_embedding_dims.items():
                self.categorical_embeddings[name] = nn.Embedding(vocab, dim)

        # Continuous projections
        self.static_continuous_transforms = nn.ModuleList(
            [nn.Linear(1, hidden_layer_size).to(device) for _ in range(num_static_continuous)])
        self.historical_continuous_transforms = nn.ModuleList(
            [nn.Linear(1, hidden_layer_size).to(device) for _ in range(num_historical_continuous)])
        self.future_continuous_transforms = nn.ModuleList(
            [nn.Linear(1, hidden_layer_size).to(device) for _ in range(num_future_continuous)])

        # Variable selection
        has_static = num_static_categorical + num_static_continuous > 0
        self.static_variable_selection = (
            VariableSelectionStatic(hidden_layer_size, dropout_rate, device) if has_static else None
        )
        has_hist = num_historical_categorical + num_historical_continuous > 0
        self.historical_variable_selection = (
            VariableSelectionTemporal(hidden_layer_size, static_context=True, dropout_rate=dropout_rate, device=device)
            if has_hist else None
        )
        has_future = num_future_categorical + num_future_continuous > 0
        self.future_variable_selection = (
            VariableSelectionTemporal(hidden_layer_size, static_context=True, dropout_rate=dropout_rate, device=device)
            if has_future else None
        )

        # Context, LSTM, enrichment, attention
        self.static_context_module = StaticContexts(hidden_layer_size, dropout_rate, device)
        self.lstm_layer = LSTMLayer(hidden_layer_size, device, num_lstm_layers, dropout_rate)
        self.static_enrichment = StaticEnrichmentLayer(hidden_layer_size, context=True, dropout_rate=dropout_rate, device=device)
        self.attention_stack = AttentionStack(num_attention_layers, hidden_layer_size, num_attention_heads, dropout_rate, device)
        self.final_gating = FinalGatingLayer(hidden_layer_size, device, dropout_rate)
        self.output_feed_forward = TFTGRNLayer(device=device, hidden_layer_size=hidden_layer_size,
                                               dropout_rate=dropout_rate, use_time_distributed=True)
        self.output_layers = nn.ModuleList(
            [nn.Linear(hidden_layer_size, 1).to(device) for _ in range(num_outputs)])

        self.to(device)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_categorical(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        if var_name not in self.categorical_embeddings:
            vocab = max(x.max().item() + 2, 10)
            self.categorical_embeddings[var_name] = nn.Embedding(int(vocab), self.hidden_layer_size,
                                                                  padding_idx=int(vocab) - 1).to(self.device)
        emb = self.categorical_embeddings[var_name]
        vocab_size = emb.num_embeddings
        x_safe = x.clone()
        x_safe[(x_safe == -1) | (x_safe >= vocab_size)] = vocab_size - 1
        x_safe = torch.clamp(x_safe, 0, vocab_size - 1)
        return emb(x_safe)

    def _prepare_static_inputs(self, static_categorical, static_continuous):
        inputs = [self._embed_categorical(c, f"static_cat_{i}") for i, c in enumerate(static_categorical)]
        for i, c in enumerate(static_continuous):
            if c.dim() == 1:
                c = c.unsqueeze(-1)
            inputs.append(self.static_continuous_transforms[i](c))
        return inputs

    def _prepare_temporal_inputs(self, cat_vars, cont_vars, transforms, prefix):
        inputs = [self._embed_categorical(c, f"{prefix}_cat_{i}") for i, c in enumerate(cat_vars)]
        for i, c in enumerate(cont_vars):
            if c.dim() == 2:
                c = c.unsqueeze(-1)
            inputs.append(apply_time_distributed(transforms[i], c))
        return inputs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        static_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        future_categorical: Optional[List[torch.Tensor]] = None,
        future_continuous: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            ``predictions`` [B, prediction_steps, num_outputs]
            ``temporal_features``, ``attention_output``,
            ``static_weights`` (if static inputs provided),
            ``historical_weights``, ``future_weights``
        """
        outputs: Dict = {}

        # Batch size
        batch_size = None
        for lst in (static_categorical, static_continuous, historical_categorical,
                    historical_continuous, future_categorical, future_continuous):
            if lst:
                batch_size = lst[0].shape[0]
                break

        # 1. Static inputs
        if self.static_variable_selection is not None:
            static_inputs = self._prepare_static_inputs(static_categorical or [], static_continuous or [])
            static_vec, sw = self.static_variable_selection(static_inputs)
            outputs['static_weights'] = sw
        else:
            static_vec = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)

        # 2. Static contexts
        stat_vec, enrich_vec, h_vec, c_vec = self.static_context_module(static_vec)

        # 3. Historical features
        if self.historical_variable_selection is not None:
            hist_inputs = self._prepare_temporal_inputs(
                historical_categorical or [], historical_continuous or [],
                self.historical_continuous_transforms, 'historical')
            hist_features, hw = self.historical_variable_selection((hist_inputs, stat_vec))
            outputs['historical_weights'] = hw
        else:
            hist_features = torch.zeros(batch_size, self.historical_steps, self.hidden_layer_size).to(self.device)

        # 4. Future features
        if self.future_variable_selection is not None:
            fut_inputs = self._prepare_temporal_inputs(
                future_categorical or [], future_continuous or [],
                self.future_continuous_transforms, 'future')
            fut_features, fw = self.future_variable_selection((fut_inputs, stat_vec))
            outputs['future_weights'] = fw
        else:
            fut_features = torch.zeros(batch_size, self.prediction_steps, self.hidden_layer_size).to(self.device)

        # 5. LSTM
        temporal_features = self.lstm_layer((hist_features, fut_features, (h_vec, c_vec)))

        # 6. Static enrichment
        enriched = self.static_enrichment((temporal_features, enrich_vec))

        # 7. Attention
        total_len = self.historical_steps + self.prediction_steps
        cm = causal_mask(total_len).to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
        attn_out = self.attention_stack(enriched, cm, padding_mask)

        # 8. Final gating + GRN
        gated = self.final_gating([attn_out, temporal_features])
        transformed = self.output_feed_forward(gated)

        # 9. Output projection (last prediction_steps only)
        pred_out = transformed[:, -self.prediction_steps:, :]
        preds = torch.cat([apply_time_distributed(self.output_layers[i], pred_out)
                           for i in range(self.num_outputs)], dim=-1)
        outputs['predictions'] = preds
        outputs['temporal_features'] = temporal_features
        outputs['attention_output'] = attn_out
        return outputs

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Convenience wrapper returning only the ``predictions`` tensor."""
        return self.forward(*args, **kwargs)['predictions']

    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                      quantiles: List[float]) -> torch.Tensor:
        """Built-in quantile loss helper."""
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        losses = []
        for i, q in enumerate(quantiles):
            e = targets - predictions[..., i:i+1]
            losses.append(torch.where(e >= 0, q * e, (q - 1) * e).mean())
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Encoder-only TFT
# ---------------------------------------------------------------------------

class TFTEncoderOnly(nn.Module):
    """
    Encoder-only Temporal Fusion Transformer.

    Use this when **no future inputs are available** and the task is to map
    a historical window to a single scalar/vector output (regression) or to
    a class label (classification).

    Parameters
    ----------
    hidden_layer_size, num_attention_heads, num_lstm_layers,
    num_attention_layers, dropout_rate : same as TemporalFusionTransformer
    num_static_categorical, num_static_continuous : int
    num_historical_categorical, num_historical_continuous : int
    categorical_embedding_dims : dict, optional
    historical_steps : int
    output_size : int
        Number of regression outputs (ignored for classification).
    output_type : str
        ``'regression'`` or ``'classification'``
    num_classes : int, optional
        Required when output_type == 'classification'.
    temporal_aggregation : str
        How to reduce the temporal sequence: ``'mean'``, ``'max'``, ``'last'``,
        or ``'attention'`` (learnable weighted sum).
    device : str
    """

    def __init__(
        self,
        hidden_layer_size: int = 160,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 1,
        num_attention_layers: int = 1,
        dropout_rate: float = 0.1,
        num_static_categorical: int = 0,
        num_static_continuous: int = 0,
        num_historical_categorical: int = 0,
        num_historical_continuous: int = 0,
        categorical_embedding_dims: Optional[Dict] = None,
        historical_steps: int = 30,
        output_size: int = 1,
        output_type: str = 'regression',
        num_classes: Optional[int] = None,
        temporal_aggregation: str = 'attention',
        device: str = 'cpu',
    ):
        super().__init__()

        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_lstm_layers = num_lstm_layers
        self.historical_steps = historical_steps
        self.output_size = output_size
        self.output_type = output_type
        self.num_classes = num_classes
        self.temporal_aggregation = temporal_aggregation

        # Embeddings
        self.categorical_embeddings = nn.ModuleDict()
        if categorical_embedding_dims:
            for name, (vocab, dim) in categorical_embedding_dims.items():
                self.categorical_embeddings[name] = nn.Embedding(vocab, dim).to(device)

        # Continuous projections
        self.static_continuous_transforms = nn.ModuleList(
            [nn.Linear(1, hidden_layer_size).to(device) for _ in range(num_static_continuous)])
        self.historical_continuous_transforms = nn.ModuleList(
            [nn.Linear(1, hidden_layer_size).to(device) for _ in range(num_historical_continuous)])

        # Variable selection
        has_static = num_static_categorical + num_static_continuous > 0
        self.static_variable_selection = (
            VariableSelectionStatic(hidden_layer_size, dropout_rate, device) if has_static else None
        )
        has_hist = num_historical_categorical + num_historical_continuous > 0
        self.historical_variable_selection = (
            VariableSelectionTemporal(hidden_layer_size, static_context=True, dropout_rate=dropout_rate, device=device)
            if has_hist else None
        )

        # Static contexts (optional)
        self.static_context_module = (
            StaticContexts(hidden_layer_size, dropout_rate, device) if has_static else None
        )

        # LSTM encoder only
        self.lstm_encoder = nn.LSTM(
            hidden_layer_size, hidden_layer_size, num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
        ).to(device)

        self.post_lstm_grn = TFTGRNLayer(device=device, hidden_layer_size=hidden_layer_size,
                                         dropout_rate=dropout_rate, use_time_distributed=True)

        self.static_enrichment = (
            StaticEnrichmentLayer(hidden_layer_size, context=True, dropout_rate=dropout_rate, device=device)
            if has_static else None
        )

        self.attention_stack = AttentionStack(num_attention_layers, hidden_layer_size,
                                              num_attention_heads, dropout_rate, device)

        # Temporal aggregation
        if temporal_aggregation == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.Tanh(),
                nn.Linear(hidden_layer_size, 1),
                nn.Softmax(dim=1),
            ).to(device)
        else:
            self.temporal_attention = None

        self.final_grn = TFTGRNLayer(device=device, hidden_layer_size=hidden_layer_size,
                                     dropout_rate=dropout_rate, use_time_distributed=False)

        out_classes = num_classes if output_type == 'classification' else output_size
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, out_classes),
        ).to(device)

        self.to(device)

    def _embed_categorical(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        if var_name not in self.categorical_embeddings:
            vocab = max(x.max().item() + 2, 10)
            self.categorical_embeddings[var_name] = nn.Embedding(
                int(vocab), self.hidden_layer_size, padding_idx=int(vocab) - 1).to(self.device)
        emb = self.categorical_embeddings[var_name]
        vocab_size = emb.num_embeddings
        x_safe = x.clone()
        x_safe[(x_safe == -1) | (x_safe >= vocab_size)] = vocab_size - 1
        x_safe = torch.clamp(x_safe, 0, vocab_size - 1)
        return emb(x_safe)

    def _prepare_static_inputs(self, static_categorical, static_continuous):
        inputs = [self._embed_categorical(c.squeeze(), f"static_cat_{i}")
                  for i, c in enumerate(static_categorical)]
        for i, c in enumerate(static_continuous):
            if c.dim() == 1:
                c = c.unsqueeze(-1)
            inputs.append(self.static_continuous_transforms[i](c))
        return inputs

    def _prepare_historical_inputs(self, cat_vars, cont_vars):
        inputs = []
        for i, c in enumerate(cat_vars):
            bs, ts = c.shape
            emb = self._embed_categorical(c.reshape(-1), f"historical_cat_{i}")
            inputs.append(emb.reshape(bs, ts, -1))
        for i, c in enumerate(cont_vars):
            if c.dim() == 2:
                c = c.unsqueeze(-1)
            inputs.append(apply_time_distributed(self.historical_continuous_transforms[i], c))
        return inputs

    def aggregate_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce [B, T, H] → [B, H]."""
        if self.temporal_aggregation == 'mean':
            return x.mean(dim=1)
        if self.temporal_aggregation == 'max':
            return x.max(dim=1)[0]
        if self.temporal_aggregation == 'last':
            return x[:, -1, :]
        # attention
        w = self.temporal_attention(x)           # [B, T, 1]
        return (x * w).sum(dim=1)

    def forward(
        self,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        static_categorical: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            ``output``  [B, output_size] or [B, num_classes]
            ``temporal_features``, ``attention_output``,
            ``historical_weights``, ``static_weights`` (when available)
            For classification: also ``logits``, ``probabilities``, ``predictions``
        """
        outputs: Dict = {}

        batch_size = None
        for lst in (historical_continuous, historical_categorical, static_continuous, static_categorical):
            if lst:
                batch_size = lst[0].shape[0]
                break
        if batch_size is None:
            raise ValueError("No inputs provided")

        # Static
        stat_vec = h0 = c0 = enrich_vec = None
        if self.static_variable_selection is not None and (static_categorical or static_continuous):
            si = self._prepare_static_inputs(static_categorical or [], static_continuous or [])
            static_vec, sw = self.static_variable_selection(si)
            outputs['static_weights'] = sw
            stat_vec, enrich_vec, h0, c0 = self.static_context_module(static_vec)
        else:
            stat_vec = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
            h0 = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
            c0 = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)

        # Historical
        if self.historical_variable_selection is not None:
            hi = self._prepare_historical_inputs(historical_categorical or [], historical_continuous or [])
            hist_feats, hw = self.historical_variable_selection((hi, stat_vec))
            outputs['historical_weights'] = hw
        else:
            hist_feats = torch.zeros(batch_size, self.historical_steps, self.hidden_layer_size).to(self.device)

        # LSTM
        h0_ = h0.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        c0_ = c0.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        lstm_out, _ = self.lstm_encoder(hist_feats, (h0_, c0_))
        temporal = self.post_lstm_grn(lstm_out)
        outputs['temporal_features'] = temporal

        # Static enrichment
        enriched = (self.static_enrichment((temporal, enrich_vec))
                    if self.static_enrichment is not None and enrich_vec is not None
                    else temporal)

        # Attention (no causal mask for encoder-only)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
        attn_out = self.attention_stack(enriched, attn_mask=None, padding_mask=padding_mask)
        outputs['attention_output'] = attn_out

        # Aggregate → output
        agg = self.aggregate_temporal_features(attn_out)
        final = self.final_grn(agg)
        out = self.output_layer(final)
        outputs['output'] = out

        if self.output_type == 'classification':
            outputs['logits'] = out
            outputs['probabilities'] = F.softmax(out, dim=-1)
            outputs['predictions'] = torch.argmax(out, dim=-1)

        return outputs

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Return predicted classes (classification) or raw output (regression)."""
        outs = self.forward(*args, **kwargs)
        return outs['predictions'] if self.output_type == 'classification' else outs['output']

    def get_feature_importance(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Return variable-selection weights as a feature-importance proxy."""
        outs = self.forward(*args, **kwargs)
        importance: Dict = {}
        if 'static_weights' in outs:
            importance['static'] = outs['static_weights']
        if 'historical_weights' in outs:
            importance['historical'] = outs['historical_weights']
        return importance
