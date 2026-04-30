#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# Helper: Imitate Keras "TimeDistributed" in PyTorch

def apply_time_distributed(module, x):
    """
    Imitate tf.keras.layers.TimeDistributed by:
      1) Flattening [B, T, ...] -> [B*T, ...]
      2) Applying the module
      3) Reshaping back to [B, T, ...]
    """
    if x.ndimension() <= 2:
        # e.g., [B, features], just apply
        return module(x)

    b, t, *others = x.size()
    #x_reshaped = x.view(b * t, *others)  # Flatten out the time dimension
    x_reshaped = x.reshape(b*t, *others)  #
    y = module(x_reshaped)
    if y.ndimension() == 2:
        # [B*T, out_dim] -> [B, T, out_dim]
        #y = y.view(b, t, y.size(-1))
        y = y.reshape(b, t, y.size(-1))
    else:
        # If there's an extra dimension, adjust accordingly
        #y = y.view(b, t, *y.shape[1:])
        y = y.reshape(b, t, *y.shape[1:])
    return y


# Masking Utils

def create_padding_mask(seq):
    """
    Imitates:
      seq = tf.cast(tf.math.less(seq, 0), tf.float32)
      return seq[:, tf.newaxis, :]  # shape (batch_size, 1, seq_len)

    For simplicity, we assume you pass in a torch.Tensor `seq`.
    We'll produce shape [B, 1, seq_len], with 1.0 where seq < 0, else 0.
    """
    mask = (seq < 0).float()  # 1 if seq < 0, else 0
    # expand dims -> [B, 1, seq_len]
    return mask.unsqueeze(1)


def causal_mask(size):
    """
    Imitates:
      mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
      return mask  # shape (seq_len, seq_len), upper-triangular 1s
    """
    # create [size, size] with 1's above the diagonal
    # PyTorch: we can use torch.triu
    mask = torch.ones(size, size).triu(diagonal=1)
    # This mask has 0 on diagonal+below, 1 above diag => "causal" means we block future tokens
    return mask


# Scaled Dot-Product Attention

def scaled_dot_product_attention(q, k, v, causal_mask=None, padding_mask=None):
    """
    Equivalent to the TF version:
      matmul_qk = tf.matmul(q, k, transpose_b=True)
      ...
      if causal_mask is not None:
          scaled_attention_logits += (causal_mask * -1e9)
      if padding_mask is not None:
          scaled_attention_logits += (padding_mask * -1e9)
      ...
      attention_weights = tf.nn.softmax(...)
      output = tf.matmul(attention_weights, v)
    """
    # q, k, v shapes: [..., seq_len_q, depth], [..., seq_len_k, depth]
    # We'll assume the leading dimensions (e.g. batch, heads) are already handled.
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = k.size(-1)
    scaled_logits = matmul_qk / (dk ** 0.5)

    #print("matmul_qk shape: ", matmul_qk.shape)
    #print("dk: ", dk)

    # Broadcast / add masks
    if causal_mask is not None:
        # causal_mask shape: [seq_len_q, seq_len_k] or broadcastable
        # We assume 1 where we should block, else 0 => multiply by -1e9
        scaled_logits = scaled_logits + (causal_mask * -1e9)

    if padding_mask is not None:
        # padding_mask shape: e.g. [B, 1, seq_len_k], must be broadcastable to scaled_logits
        scaled_logits = scaled_logits + (padding_mask * -1e9)

    # Softmax across seq_len_k dimension (the last dimension)
    attention_weights = F.softmax(scaled_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    #print("scaled dot prod attn output: ", output.shape)

    return output, attention_weights


# In[ ]:


# Multi-Head Attention

class TFTMultiHeadAttention(nn.Module):
    """
    PyTorch equivalent of your 'TFTMultiHeadAttention' layer.
    """

    def __init__(self, n_head, d_model, device, dropout_rate=0.1):
        """
        n_head: number of attention heads
        d_model: total hidden dimension (must be divisible by n_head)
        """
        super().__init__()

        self.device = device
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_k = self.d_v = d_model // n_head
        self.dropout_rate = dropout_rate

        # We replicate your code: same vs_layer across all heads
        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(self.device)
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(self.device)
        # One vs_layer shared by all heads
        vs_layer = nn.Linear(d_model, self.d_v, bias=False).to(self.device)
        self.vs_layers = nn.ModuleList([vs_layer for _ in range(n_head)])

        self.dropout = nn.Dropout(dropout_rate)
        self.w_o = nn.Linear(d_model, d_model, bias=False).to(self.device)

    def forward(self, q, k, v, causal_mask=None, padding_mask=None):
        """
        q, k, v: [batch, seq_len, d_model]
        causal_mask: [seq_len, seq_len] or broadcastable
        padding_mask: [batch, 1, seq_len] or broadcastable
        return: (outputs, attn)
          outputs: [batch, seq_len, d_model]
          attn: [n_head, batch, seq_len_q, seq_len_k]
        """
        heads = []
        attns = []

        for i in range(self.n_head):
            qs = self.qs_layers[i](q)  # [B, T, d_k]
            ks = self.ks_layers[i](k)  # [B, T, d_k]
            vs = self.vs_layers[i](v)  # [B, T, d_v]

            head, attn = scaled_dot_product_attention(qs, ks, vs, causal_mask, padding_mask)
            head = self.dropout(head)  # dropout on each head
            heads.append(head)
            attns.append(attn)

        # Stack heads => shape [n_head, B, T, d_v]
        #head = torch.stack(heads, dim=0)  # (n_head, B, T, d_v)
        
        out = torch.cat(heads, dim=-1)
        
        attn = torch.stack(attns, dim=0)  # (n_head, B, T, T)
        
        #print("head : ", head.shape)
        #print("attn : ", attn.shape)
        # Average across heads => shape [B, T, d_v]
        # ( or you could concatenate them if you prefer the standard MHA practice,
        #   but your code does reduce_mean across heads)
        #if self.n_head > 1:
        #    out = torch.mean(head, dim=0)  # [B, T, d_v]
        #else:
        #    out = head.squeeze(0)  # [B, T, d_v]

        out = self.w_o(out)  # project back to d_model
        out = self.dropout(out)

        return out, attn
    


# In[ ]:


# Other Temporal Fusion Transformer Components

# Gated Residual Network (GRN) + Sub-layers

def get_activation_fn(activation_str):
    """
    Utility to map string to PyTorch activation function
    """
    if activation_str is None:
        return None
    activation_str = activation_str.lower()
    if activation_str == 'elu':
        return F.elu
    elif activation_str == 'tanh':
        return torch.tanh
    elif activation_str == 'sigmoid':
        return torch.sigmoid
    elif activation_str == 'softmax':
        return F.softmax
    # Add more as needed
    else:
        raise ValueError(f"Unsupported activation: {activation_str}")


class TFTLinearLayer(nn.Module):
    """
    Equivalent to 'tft_linear_layer' in Keras code:
    A Dense layer with optional time-distribution and optional activation.
    """

    def __init__(self, hidden_layer_size, device, activation=None, use_time_distributed=False, use_bias=True):
        super().__init__()
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.layer = nn.Linear(in_features=0, out_features=0, bias=use_bias).to(self.device)
        # Will set in_features dynamically at forward if desired (or pass it in the init).
        # For simplicity, we assume you know the in_features at construction time in real usage.

        self.hidden_layer_size = hidden_layer_size
        self.activation_fn = get_activation_fn(activation)

        # We will build the linear layer on-the-fly once we see the input dimension
        # if you need strict initialization, you can do so after knowing input_dim.

    def forward(self, x):
        # If we haven't set the in_features/out_features yet, do so here
        if self.layer.in_features == 0:
            # x shape: [B, T, in_features] or [B, in_features]
            if x.ndimension() == 3:
                # [B, T, F]
                in_features = x.shape[-1]
            else:
                # [B, F]
                in_features = x.shape[-1]
            self.layer = nn.Linear(in_features, self.hidden_layer_size, bias=self.layer.bias is not None).to(self.device)

        if self.use_time_distributed:
            out = apply_time_distributed(self.layer, x)
        else:
            out = self.layer(x)

        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class TFTApplyMLP(nn.Module):
    """
    Equivalent to 'tft_apply_mlp'.
    """

    def __init__(self, hidden_size, output_size, output_activation=None, hidden_activation='tanh',
                 use_time_distributed=False):
        super().__init__()
        self.use_time_distributed = use_time_distributed

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        if hidden_activation.lower() == "tanh":
            self.hidden_activation_fn = torch.tanh
        elif hidden_activation.lower() == "elu":
            self.hidden_activation_fn = F.elu
        else:
            raise ValueError(f"Unsupported activation {hidden_activation}")

        # Output layer
        self.out_layer = nn.Linear(hidden_size, output_size)
        if output_activation is None:
            self.out_activation_fn = None
        elif output_activation.lower() == "sigmoid":
            self.out_activation_fn = torch.sigmoid
        elif output_activation.lower() == "softmax":
            self.out_activation_fn = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported activation {output_activation}")

    def forward(self, x):
        if self.use_time_distributed:
            x = apply_time_distributed(self.hidden_layer, x)
            x = self.hidden_activation_fn(x)
            x = apply_time_distributed(self.out_layer, x)
            if self.out_activation_fn is not None:
                x = self.out_activation_fn(x)
        else:
            x = self.hidden_layer(x)
            x = self.hidden_activation_fn(x)
            x = self.out_layer(x)
            if self.out_activation_fn is not None:
                x = self.out_activation_fn(x)
        return x


class TFTApplyGatingLayer(nn.Module):
    """
    Equivalent to 'tft_apply_gating_layer'.
    """

    def __init__(self, hidden_layer_size, device, dropout_rate=0.0, use_time_distributed=True, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.hidden_layer_size = hidden_layer_size

        self.activation_fc = nn.Linear(in_features=0, out_features=0).to(self.device)
        self.gated_fc = nn.Linear(in_features=0, out_features=0).to(self.device)

        if activation is None:
            self.activation_fn = None
        elif activation.lower() == "elu":
            self.activation_fn = F.elu
        # Add other activations as desired
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x):
        # If we haven't set the in_features/out_features yet, do so here
        if self.activation_fc.in_features == 0:
            # x shape: [B, T, in_features] or [B, in_features]
            if x.ndimension() == 3:
                # [B, T, F]
                in_features = x.shape[-1]
            else:
                # [B, F]
                in_features = x.shape[-1]
            self.activation_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
            self.gated_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)

        x = self.dropout(x)
        if self.use_time_distributed:
            a = apply_time_distributed(self.activation_fc, x)
            g = apply_time_distributed(self.gated_fc, x)
        else:
            a = self.activation_fc(x)
            g = self.gated_fc(x)

        if self.activation_fn is not None:
            a = self.activation_fn(a)
        g = torch.sigmoid(g)  # gating uses sigmoid

        out = a * g
        return out, g


class TFTAddAndNormLayer(nn.Module):
    """
    Equivalent to 'tft_add_and_norm_layer'.
    """

    def __init__(self, device, normalized_shape):
        super().__init__()
        self.device = device
        self.layer_norm = nn.LayerNorm(normalized_shape).to(device) #None  # We'll build dynamically based on last dimension

    def forward(self, inputs):
        # inputs is a list or tuple: [skip, gating_layer]
        skip, gating_layer = inputs
        out = skip + gating_layer

        #if self.layer_norm is None:
        #    # Build LayerNorm if needed, expecting to normalize over last dim
        #    norm_shape = out.shape[-1]
        #    self.layer_norm = nn.LayerNorm(norm_shape).to(self.device)

        out = self.layer_norm(out)
        return out


class TFTGRNLayer(nn.Module):
    """
    Equivalent to 'tft_grn_layer'.
    """

    def __init__(self,
                 device,
                 hidden_layer_size,
                 output_size=None,
                 dropout_rate=0.0,
                 use_time_distributed=True,
                 additional_context=False,
                 return_gate=False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size if output_size is not None else hidden_layer_size
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        self.return_gate = return_gate
        self.device = device

        # Main linear (skip connection)
        input_size = hidden_layer_size*self.output_size if output_size is not None else hidden_layer_size
        self.skip_layer = nn.Linear(input_size, self.output_size)

        # Hidden layers
        self.hidden_1 = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                       activation=None,
                                       use_time_distributed=use_time_distributed,
                                       device=self.device)

        self.hidden_2 = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                       activation=None,
                                       use_time_distributed=use_time_distributed,
                                       device=self.device)

        if additional_context:
            self.context_layer = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                                activation=None,
                                                use_time_distributed=use_time_distributed,
                                                device=self.device,
                                                use_bias=False)
        else:
            self.context_layer = None

        # Gate
        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.output_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=use_time_distributed,
                                        activation=None,
                                        device=self.device)

        # Add & Norm
        self.add_norm = TFTAddAndNormLayer(device=self.device, normalized_shape=self.output_size)

    def forward(self, inputs):
        """
        If additional_context == True, inputs is (x, c)
        else inputs is just x
        """
        if self.additional_context:
            x, c = inputs
        else:
            x = inputs

        # skip
        skip = x
        # If skip dimension != output_size,
        # you might need another linear or shape transform.
        # Here we assume x.shape[-1] == hidden_layer_size
        # and skip_layer out_features == self.output_size.

        # Convert skip to [B, T, output_size] if time_distributed
        if self.use_time_distributed:
            skip_out = apply_time_distributed(self.skip_layer, skip)
        else:
            skip_out = self.skip_layer(skip)

        # hidden path
        h = self.hidden_1(x)
        if self.context_layer is not None:
            # broadcast c if necessary
            if h.ndimension() == 3 and c.ndimension() == 2:
                # expand c to [B, 1, C]
                c = c.unsqueeze(1)
            c_out = self.context_layer(c)
            h = h + c_out

        h = F.elu(h)
        h = self.hidden_2(h)

        gating_layer, gate = self.gate(h)
        out = self.add_norm([skip_out, gating_layer])

        if self.return_gate:
            return out, gate
        return out


# Static Variable Selection

class VariableSelectionStatic(nn.Module):
    """
    Equivalent to 'variable_selection_static'.
    """

    def __init__(self, hidden_layer_size, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        # We will build sub-layers once we know num_vars
        self.num_vars = None
        self.grn_flat = None
        self.grn_vars = nn.ModuleList()
        self.device = device

    def build_layers(self, num_vars):
        self.num_vars = num_vars
        self.grn_flat = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                    output_size=self.num_vars,
                                    dropout_rate=self.dropout_rate,
                                    use_time_distributed=False,
                                    additional_context=False,
                                    return_gate=False,
                                    device=self.device).to(self.device)

        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                            output_size=None,
                            dropout_rate=self.dropout_rate,
                            use_time_distributed=False,
                            additional_context=False,
                            return_gate=False,
                            device=self.device).to(self.device)
            )

    def forward(self, inputs):
        """
        inputs: list of static vars, each shape [B, var_dim].
        We concatenate => [B, sum_of_var_dims].
        """
        if self.grn_flat is None:
            self.build_layers(num_vars=len(inputs))

        flatten = torch.cat(inputs, dim=1)  # [B, sum_of_var_dims]
        mlp_outputs = self.grn_flat(flatten)  # [B, num_vars]

        # Softmax over num_vars dimension
        static_weights = F.softmax(mlp_outputs, dim=-1)  # [B, num_vars]
        weights = static_weights.unsqueeze(-1)  # [B, num_vars, 1]

        # Transform each var
        transformed = []
        for i in range(self.num_vars):
            e = self.grn_vars[i](inputs[i])  # [B, hidden_layer_size]
            transformed.append(e)

        # Stack => [B, num_vars, hidden_layer_size]
        trans_embedding = torch.stack(transformed, dim=1)

        # Weighted sum
        combined = trans_embedding * weights  # broadcast
        static_vec = combined.sum(dim=1)  # [B, hidden_layer_size]

        return static_vec, static_weights


# Static Contexts

class StaticContexts(nn.Module):
    """
    Equivalent to 'static_contexts'.
    """

    def __init__(self, hidden_layer_size, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.stat_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                          output_size=None,
                                          dropout_rate=self.dropout_rate,
                                          use_time_distributed=False,
                                          additional_context=False,
                                          return_gate=False,
                                          device=self.device)

        self.enrich_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                            output_size=None,
                                            dropout_rate=self.dropout_rate,
                                            use_time_distributed=False,
                                            additional_context=False,
                                            return_gate=False,
                                            device=self.device)

        self.h_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                       output_size=None,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=False,
                                       additional_context=False,
                                       return_gate=False,
                                       device=self.device)

        self.c_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                       output_size=None,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=False,
                                       additional_context=False,
                                       return_gate=False,
                                       device=self.device)

    def forward(self, inputs):
        # inputs: static_vec => [B, hidden_layer_size]
        stat_vec = self.stat_vec_layer(inputs)
        enrich_vec = self.enrich_vec_layer(inputs)
        h_vec = self.h_vec_layer(inputs)
        c_vec = self.c_vec_layer(inputs)

        return stat_vec, enrich_vec, h_vec, c_vec

# Temporal Variable Selection

class VariableSelectionTemporal(nn.Module):
    """
    Equivalent to 'variable_selection_temporal'.
    Takes inputs as a list of [B, T, dim] plus a context vector if `static_context=True`.
    """

    def __init__(self, hidden_layer_size, static_context, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.static_context = static_context
        self.dropout_rate = dropout_rate
        # We'll build on first forward pass
        self.num_vars = None
        self.grn_flat = None
        self.grn_vars = nn.ModuleList()
        self.device = device

    def build_layers(self, num_vars):
        self.num_vars = num_vars
        self.grn_flat = TFTGRNLayer(
            hidden_layer_size=self.hidden_layer_size,
            output_size=self.num_vars,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=self.static_context,
            return_gate=False,
            device=self.device
        ).to(self.device)

        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                            output_size=None,
                            dropout_rate=self.dropout_rate,
                            use_time_distributed=True,
                            additional_context=False,
                            return_gate=False,
                            device=self.device).to(self.device)
            )

    def forward(self, x):
        """
        x:
         - if static_context=True, (list_of_tensors, context)
         - else, list_of_tensors
        list_of_tensors: each shape [B, T, dim_i]
        context: shape [B, context_dim]
        """
        if self.static_context:
            inputs, context = x
            if self.grn_flat is None:
                self.build_layers(len(inputs))

            # Expand context => [B, 1, context_dim]
            context = context.unsqueeze(1)
            flatten = torch.cat(inputs, dim=-1)  # [B, T, sum_of_dims]
            mlp_outputs = self.grn_flat((flatten, context))
        else:
            inputs = x
            if self.grn_flat is None:
                self.build_layers(len(inputs))
            flatten = torch.cat(inputs, dim=-1)  # [B, T, sum_of_dims]
            mlp_outputs = self.grn_flat(flatten)

        # TimeDistributed(Activation('softmax')) => apply softmax over last dim => num_vars
        dynamic_weights = F.softmax(mlp_outputs, dim=-1)  # [B, T, num_vars]
        weights = dynamic_weights.unsqueeze(-1)  # [B, T, num_vars, 1]

        # Transform each variable
        transformed = []
        for i in range(self.num_vars):
            e = self.grn_vars[i](inputs[i])  # [B, T, hidden_size]
            transformed.append(e)

        # [B, T, num_vars, hidden_size]
        trans_embedding = torch.stack(transformed, dim=2)

        # Weighted sum across num_vars
        combined = weights * trans_embedding
        lstm_input = combined.sum(dim=2)  # [B, T, hidden_size]

        return lstm_input, dynamic_weights

    
# LSTM Layer

class LSTMLayer(nn.Module):
    """
    Equivalent to 'lstm_layer' in your code.
    """

    def __init__(self, hidden_layer_size, device, rnn_layers, dropout_rate):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        # Single-layer LSTM for encoder & decoder.
        # If you want multiple layers, you can expand to 'num_layers=rnn_layers'.
        self.tft_encoder = nn.LSTM(input_size=hidden_layer_size,
                                   hidden_size=hidden_layer_size,
                                   num_layers=rnn_layers,
                                   batch_first=True)
        self.tft_decoder = nn.LSTM(input_size=hidden_layer_size,
                                   hidden_size=hidden_layer_size,
                                   num_layers=rnn_layers,
                                   batch_first=True)

        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.hidden_layer_size,
                                        dropout_rate=self.dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)

        self.add_norm = TFTAddAndNormLayer(device=self.device, normalized_shape=hidden_layer_size)

    def forward(self, inputs):
        """
        inputs: (encoder_in, decoder_in, init_states)
          encoder_in: [B, T_enc, hidden_layer_size]
          decoder_in: [B, T_dec, hidden_layer_size]
          init_states: tuple (h0, c0) each shape [B, hidden_layer_size]
                       (for a single-layer LSTM).
        """
        encoder_in, decoder_in, init_states = inputs
        # PyTorch LSTM expects states in shape (num_layers, B, hidden_size).
        # So we must expand dims: [B, hidden_size] -> [1, B, hidden_size]
        h0, c0 = init_states
        h0 = h0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)

        # Run encoder
        encoder_out, (enc_h, enc_c) = self.tft_encoder(encoder_in, (h0, c0))
        # Run decoder, using final encoder states
        decoder_out, _ = self.tft_decoder(decoder_in, (enc_h, enc_c))

        # Concat => [B, T_enc+T_dec, hidden_layer_size]
        lstm_out = torch.cat([encoder_out, decoder_out], dim=1)
        # Residual skip input
        lstm_in = torch.cat([encoder_in, decoder_in], dim=1)

        # Apply gating
        # gating expects shape [B, T, hidden_size]; we do time-distributed gating
        out, _ = self.gate(lstm_out)

        # Add & norm
        temporal_features = self.add_norm([out, lstm_in])
        return temporal_features


# Static Enrichment Layer

class StaticEnrichmentLayer(nn.Module):
    """
    Equivalent to 'static_enrichment_layer'.
    """

    def __init__(self, hidden_layer_size, context, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.context = context
        self.dropout_rate = dropout_rate
        self.device = device

        self.grn_enrich = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                      output_size=None,
                                      dropout_rate=self.dropout_rate,
                                      use_time_distributed=True,
                                      additional_context=self.context,
                                      return_gate=False,
                                      device=self.device)

    def forward(self, inputs):
        """
        inputs:
          if self.context=True, (temporal_features, static_enrichment_vec)
          else, just temporal_features
        shapes:
          temporal_features => [B, T, hidden_size]
          static_enrichment_vec => [B, context_dim]
        """
        if self.context:
            x, c = inputs
            # expand c => [B, 1, context_dim]
            c = c.unsqueeze(1)
            enriched = self.grn_enrich((x, c))
        else:
            x = inputs
            enriched = self.grn_enrich(x)

        return enriched


# Attention Layer & Stack

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, n_head, dropout_rate):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.mha = TFTMultiHeadAttention(n_head=n_head, d_model=hidden_layer_size, dropout_rate=dropout_rate, device=self.device)
        self.gate = TFTApplyGatingLayer(hidden_layer_size=hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)
        self.add_norm = TFTAddAndNormLayer(device=self.device, normalized_shape=hidden_layer_size)

    def forward(self, x, attn_mask, padding_mask):

        attn_out, _ = self.mha(x, x, x, attn_mask, padding_mask) # (q,k,v,mask,training)

        # gating layer
        attn_out, _ = self.gate(attn_out)
        # add_norm
        attn_out = self.add_norm([attn_out, x])
        return attn_out


# Attention Stack
class AttentionStack(nn.Module):
    def __init__(self, num_layers, hidden_layer_size, n_head, dropout_rate, device):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        #self.attn_layers = [AttentionLayer(hidden_layer_size, device, n_head, dropout_rate) for _ in range(num_layers)]
        self.attn_layers = nn.ModuleList([AttentionLayer(hidden_layer_size, device, n_head, dropout_rate) for _ in range(num_layers)])
        self.device = device
        self.grn_final = TFTGRNLayer(hidden_layer_size=hidden_layer_size,
                                     output_size=None,
                                     dropout_rate=dropout_rate,
                                     use_time_distributed=True,
                                     additional_context=False,
                                     return_gate=False,
                                     device=self.device)

    def forward(self, x, attn_mask, padding_mask):

        attn_out = x
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](attn_out, attn_mask, padding_mask)

        # final GRN layer
        attn_out = self.grn_final(attn_out)
        return attn_out


# Final Gating Layer

class FinalGatingLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, dropout_rate):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.gate = TFTApplyGatingLayer(hidden_layer_size=hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)
        self.add_norm = TFTAddAndNormLayer(device=self.device, normalized_shape=hidden_layer_size)

    def forward(self, inputs):

        attn_out, temporal_features = inputs

        # final gating layer
        attn_out, _ = self.gate(attn_out)

        # final add & norm
        out = self.add_norm([attn_out, temporal_features])

        return out
    


# In[ ]:


# TemporalFusionTransformer Model

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) Model for time series forecasting.
    
    The TFT combines various deep learning architectures:
    - Variable selection networks for relevant feature selection
    - LSTM layers for temporal processing
    - Multi-head attention for long-range dependencies
    - Gated residual networks (GRN) for efficient information flow
    
    Architecture flow:
    1. Static and temporal variable selection
    2. Static covariate encoding
    3. Temporal processing with LSTM
    4. Static enrichment of temporal features
    5. Temporal self-attention
    6. Position-wise feed-forward
    7. Output generation with quantile forecasts
    """
    
    def __init__(
        self,
        # Model architecture parameters
        hidden_layer_size: int = 160,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 1,
        num_attention_layers: int = 1,
        dropout_rate: float = 0.1,
        
        # Input specification
        num_static_categorical: int = 0,
        num_static_continuous: int = 0,
        num_historical_categorical: int = 0,
        num_historical_continuous: int = 0,
        num_future_categorical: int = 0,
        num_future_continuous: int = 0,
        
        # Embedding dimensions for categorical variables
        categorical_embedding_dims: Optional[Dict[str, int]] = None,
        
        # Time dimensions
        historical_steps: int = 168,  # Number of historical time steps
        prediction_steps: int = 24,   # Number of future steps to predict
        
        # Output specification
        num_outputs: int = 3,  # Number of quantiles to predict
        
        # Device
        device: str = 'cpu'
    ):
        """
        Initialize the Temporal Fusion Transformer model.
        
        Args:
            hidden_layer_size: Size of hidden layers throughout the model
            num_attention_heads: Number of attention heads in multi-head attention
            num_lstm_layers: Number of LSTM layers
            num_attention_layers: Number of self-attention layers
            dropout_rate: Dropout rate for regularization
            num_static_categorical: Number of static categorical variables
            num_static_continuous: Number of static continuous variables
            num_historical_categorical: Number of historical categorical variables
            num_historical_continuous: Number of historical continuous variables
            num_future_categorical: Number of future categorical variables
            num_future_continuous: Number of future continuous variables
            categorical_embedding_dims: Dictionary mapping categorical variable names to embedding dimensions
            historical_steps: Number of historical time steps in input
            prediction_steps: Number of future time steps to predict
            num_outputs: Number of output quantiles
            device: Device to run the model on
        """
        super().__init__()
        
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_attention_heads = num_attention_heads
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_layers = num_attention_layers
        self.dropout_rate = dropout_rate
        
        self.num_static_categorical = num_static_categorical
        self.num_static_continuous = num_static_continuous
        self.num_historical_categorical = num_historical_categorical
        self.num_historical_continuous = num_historical_continuous
        self.num_future_categorical = num_future_categorical
        self.num_future_continuous = num_future_continuous
        
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.total_time_steps = historical_steps + prediction_steps
        self.num_outputs = num_outputs
        
        # Initialize embedding layers for categorical variables
        self.categorical_embeddings = nn.ModuleDict()
        if categorical_embedding_dims:
            for var_name, embed_dim in categorical_embedding_dims.items():
                # Assuming max category value is provided or use a default
                self.categorical_embeddings[var_name] = nn.Embedding(
                    num_embeddings=embed_dim[0],  # vocabulary size
                    embedding_dim=embed_dim[1]    # embedding dimension
                )
        
        # Input transformation layers for continuous variables
        self.static_continuous_transforms = nn.ModuleList([
            nn.Linear(1, hidden_layer_size).to(device) 
            for _ in range(num_static_continuous)
        ])
        
        self.historical_continuous_transforms = nn.ModuleList([
            nn.Linear(1, hidden_layer_size).to(device) 
            for _ in range(num_historical_continuous)
        ])
        
        self.future_continuous_transforms = nn.ModuleList([
            nn.Linear(1, hidden_layer_size).to(device) 
            for _ in range(num_future_continuous)
        ])
        
        # Variable Selection Networks
        if num_static_categorical + num_static_continuous > 0:
            self.static_variable_selection = VariableSelectionStatic(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                device=device
            )
            # eagerly build so params are visible to the optimizer
            self.static_variable_selection.build_layers(
                num_vars=num_static_categorical + num_static_continuous
            )
        else:
            self.static_variable_selection = None
            
        # Historical variable selection (if we have historical inputs)
        if num_historical_categorical + num_historical_continuous > 0:
            self.historical_variable_selection = VariableSelectionTemporal(
                hidden_layer_size=hidden_layer_size,
                static_context=True,  # Use static context
                dropout_rate=dropout_rate,
                device=device
            )
            # eagerly build so params are visible to the optimizer
            self.historical_variable_selection.build_layers(num_vars=num_historical_categorical + num_historical_continuous)
        else:
            self.historical_variable_selection = None
            
        # Future variable selection (if we have future/known inputs)
        if num_future_categorical + num_future_continuous > 0:
            self.future_variable_selection = VariableSelectionTemporal(
                hidden_layer_size=hidden_layer_size,
                static_context=True,  # Use static context
                dropout_rate=dropout_rate,
                device=device
            )
            # eagerly build so params are visible to the optimizer
            self.future_variable_selection.build_layers(num_vars=num_future_categorical + num_future_continuous)
        else:
            self.future_variable_selection = None
        
        # Static Context Networks
        self.static_context_module = StaticContexts(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # LSTM Layer for temporal processing
        self.lstm_layer = LSTMLayer(
            hidden_layer_size=hidden_layer_size,
            device=device,
            rnn_layers=num_lstm_layers,
            dropout_rate=dropout_rate
        )
        
        # Static Enrichment Layer
        self.static_enrichment = StaticEnrichmentLayer(
            hidden_layer_size=hidden_layer_size,
            context=True,  # Use static context for enrichment
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Self-Attention Stack
        self.attention_stack = AttentionStack(
            num_layers=num_attention_layers,
            hidden_layer_size=hidden_layer_size,
            n_head=num_attention_heads,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Final Gating Layer
        self.final_gating = FinalGatingLayer(
            hidden_layer_size=hidden_layer_size,
            device=device,
            dropout_rate=dropout_rate
        )
        
        # Position-wise feed-forward for output
        self.output_feed_forward = TFTGRNLayer(
            device=device,
            hidden_layer_size=hidden_layer_size,
            output_size=None,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            additional_context=False,
            return_gate=False
        )
        
        # Final output projection layers for quantile predictions
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_layer_size, 1).to(device) 
            for _ in range(num_outputs)
        ])
        
        self.to(device)
    
    def _embed_categorical(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        """
        Safely embed categorical variable.

        Args:
            x: Input tensor with categorical indices (may contain -1 for unknown)
            var_name: Name of the categorical variable

        Returns:
            Embedded tensor
        """
        if var_name in self.categorical_embeddings:
            embed_layer = self.categorical_embeddings[var_name]
            vocab_size = embed_layer.num_embeddings

            # Create safe indices
            x_safe = x.clone()

            # Map special values:
            # -1 (unknown) -> vocab_size - 1 (last index)
            # Any value >= vocab_size -> vocab_size - 1
            mask_unknown = (x_safe == -1)
            mask_overflow = (x_safe >= vocab_size)

            x_safe[mask_unknown] = vocab_size - 1
            x_safe[mask_overflow] = vocab_size - 1

            # Ensure all values are in valid range
            x_safe = torch.clamp(x_safe, min=0, max=vocab_size - 1)

            return embed_layer(x_safe)
        else:
            # Handle missing embedding gracefully
            # Create a learnable embedding on the fly
            max_idx = x.max().item() if x.numel() > 0 else 1
            vocab_size = max(max_idx + 2, 10)  # At least 10 for stability

            embed_layer = nn.Embedding(
                vocab_size, 
                self.hidden_layer_size,
                padding_idx=vocab_size - 1
            ).to(self.device)

            # Add to embeddings dict for future use
            self.categorical_embeddings[var_name] = embed_layer

            # Recursively call with the new embedding
            return self._embed_categorical(x, var_name)
    
    def _prepare_static_inputs(self, static_categorical: List[torch.Tensor], 
                              static_continuous: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Prepare static inputs by embedding categoricals and transforming continuous variables.
        
        Args:
            static_categorical: List of static categorical variables [batch_size, 1] each
            static_continuous: List of static continuous variables [batch_size, 1] each
            
        Returns:
            List of transformed static inputs
        """
        static_inputs = []
        
        # Process categorical variables
        for i, cat_var in enumerate(static_categorical):
            embedded = self._embed_categorical(cat_var, f"static_cat_{i}")
            static_inputs.append(embedded)
        
        # Process continuous variables
        for i, cont_var in enumerate(static_continuous):
            if cont_var.dim() == 1:
                cont_var = cont_var.unsqueeze(-1)
            transformed = self.static_continuous_transforms[i](cont_var)
            static_inputs.append(transformed)
        
        return static_inputs
    
    def _prepare_temporal_inputs(self, 
                                categorical_vars: List[torch.Tensor],
                                continuous_vars: List[torch.Tensor],
                                transforms: nn.ModuleList,
                                var_prefix: str) -> List[torch.Tensor]:
        """
        Prepare temporal inputs (historical or future).
        
        Args:
            categorical_vars: List of categorical variables [batch_size, time_steps]
            continuous_vars: List of continuous variables [batch_size, time_steps]
            transforms: List of transformation layers for continuous variables
            var_prefix: Prefix for variable names ('historical' or 'future')
            
        Returns:
            List of transformed temporal inputs [batch_size, time_steps, hidden_size]
        """
        temporal_inputs = []
        
        # Process categorical variables
        for i, cat_var in enumerate(categorical_vars):
            # cat_var shape: [batch_size, time_steps]
            embedded = self._embed_categorical(cat_var, f"{var_prefix}_cat_{i}")
            # embedded shape: [batch_size, time_steps, embed_dim]
            temporal_inputs.append(embedded)
        
        # Process continuous variables
        for i, cont_var in enumerate(continuous_vars):
            # cont_var shape: [batch_size, time_steps]
            if cont_var.dim() == 2:
                cont_var = cont_var.unsqueeze(-1)  # [batch_size, time_steps, 1]
            
            # Apply time-distributed transformation
            transformed = apply_time_distributed(transforms[i], cont_var)
            temporal_inputs.append(transformed)
        
        return temporal_inputs
    
    def forward(self, 
                static_categorical: Optional[List[torch.Tensor]] = None,
                static_continuous: Optional[List[torch.Tensor]] = None,
                historical_categorical: Optional[List[torch.Tensor]] = None,
                historical_continuous: Optional[List[torch.Tensor]] = None,
                future_categorical: Optional[List[torch.Tensor]] = None,
                future_continuous: Optional[List[torch.Tensor]] = None,
                padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Temporal Fusion Transformer.
        
        Args:
            static_categorical: List of static categorical inputs
            static_continuous: List of static continuous inputs
            historical_categorical: List of historical categorical inputs
            historical_continuous: List of historical continuous inputs
            future_categorical: List of future/known categorical inputs
            future_continuous: List of future/known continuous inputs
            padding_mask: To indicate left padded time periods 
            
        Returns:
            Dictionary containing:
                - 'predictions': Quantile predictions [batch_size, prediction_steps, num_outputs]
                - 'attention_weights': Attention weights from the model
                - 'static_weights': Static variable selection weights
                - 'historical_weights': Historical variable selection weights
                - 'future_weights': Future variable selection weights
        """
        batch_size = None
        outputs = {}
        
        # Determine batch size from first available input
        for inputs in [static_categorical, static_continuous, 
                      historical_categorical, historical_continuous,
                      future_categorical, future_continuous]:
            if inputs is not None and len(inputs) > 0:
                batch_size = inputs[0].shape[0]
                break
        
        # 1. Process static inputs and variable selection
        static_vec = None
        static_weights = None
        if self.static_variable_selection is not None:
            static_inputs = self._prepare_static_inputs(
                static_categorical or [],
                static_continuous or []
            )
            static_vec, static_weights = self.static_variable_selection(static_inputs)
            outputs['static_weights'] = static_weights
        else:
            # Create a default static vector if no static inputs
            static_vec = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
        
        # 2. Generate static contexts
        stat_vec, enrich_vec, h_vec, c_vec = self.static_context_module(static_vec)
        
        # 3. Process historical inputs with variable selection
        historical_features = None
        if self.historical_variable_selection is not None:
            historical_inputs = self._prepare_temporal_inputs(
                historical_categorical or [],
                historical_continuous or [],
                self.historical_continuous_transforms,
                'historical'
            )
            historical_features, historical_weights = self.historical_variable_selection(
                (historical_inputs, stat_vec)
            )
            outputs['historical_weights'] = historical_weights
        else:
            # Create zero historical features if no historical inputs
            historical_features = torch.zeros(
                batch_size, self.historical_steps, self.hidden_layer_size
            ).to(self.device)
        
        # 4. Process future inputs with variable selection
        future_features = None
        if self.future_variable_selection is not None:
            future_inputs = self._prepare_temporal_inputs(
                future_categorical or [],
                future_continuous or [],
                self.future_continuous_transforms,
                'future'
            )
            future_features, future_weights = self.future_variable_selection(
                (future_inputs, stat_vec)
            )
            outputs['future_weights'] = future_weights
        else:
            # Create zero future features if no future inputs
            future_features = torch.zeros(
                batch_size, self.prediction_steps, self.hidden_layer_size
            ).to(self.device)
        
        # 5. LSTM processing
        # Combine historical and future features for LSTM
        lstm_input = (historical_features, future_features, (h_vec, c_vec))
        temporal_features = self.lstm_layer(lstm_input)
        
        # 6. Static enrichment
        enriched_features = self.static_enrichment((temporal_features, enrich_vec))
        
        # 7. Self-attention processing
        # Create attention masks
        # Causal mask for decoder part (future predictions)
        total_seq_len = self.historical_steps + self.prediction_steps
        causal_attention_mask = causal_mask(total_seq_len).to(self.device)
        
        # NEW: Use the provided padding mask if available
        if padding_mask is not None:
            # Ensure correct shape and device
            padding_mask = padding_mask.to(self.device)
            # padding_mask shape should be [batch_size, 1, seq_len]
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
        else:
            padding_mask = None
        
        # Apply attention stack
        attention_output = self.attention_stack(
            enriched_features, 
            causal_attention_mask, 
            padding_mask
        )
        
        # 8. Final gating layer
        gated_output = self.final_gating([attention_output, temporal_features])
        
        # 9. Position-wise feed-forward
        transformed_output = self.output_feed_forward(gated_output)
        
        # 10. Extract prediction time steps only
        # We only want the last prediction_steps outputs
        prediction_outputs = transformed_output[:, -self.prediction_steps:, :]
        
        # 11. Generate quantile predictions
        quantile_predictions = []
        for i in range(self.num_outputs):
            # Apply time-distributed output layer
            quantile_pred = apply_time_distributed(
                self.output_layers[i], 
                prediction_outputs
            )
            quantile_predictions.append(quantile_pred)
        
        # Stack quantile predictions: [batch_size, prediction_steps, num_quantiles]
        predictions = torch.cat(quantile_predictions, dim=-1)
        outputs['predictions'] = predictions
        
        # Store attention weights if needed
        outputs['temporal_features'] = temporal_features
        outputs['attention_output'] = attention_output
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights tensor
        """
        # This would need to be implemented based on storing attention weights
        # during forward pass in the attention layers
        pass
    
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        Convenience method for prediction that returns only the predictions.
        
        Returns:
            Prediction tensor [batch_size, prediction_steps, num_outputs]
        """
        outputs = self.forward(*args, **kwargs)
        return outputs['predictions']
    
    def quantile_loss(self, predictions: torch.Tensor, 
                     targets: torch.Tensor, 
                     quantiles: List[float]) -> torch.Tensor:
        """
        Calculate quantile loss for training.
        
        Args:
            predictions: Model predictions [batch_size, time_steps, num_quantiles]
            targets: Target values [batch_size, time_steps]
            quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
            
        Returns:
            Quantile loss value
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # [batch_size, time_steps, 1]
        
        losses = []
        for i, q in enumerate(quantiles):
            pred_q = predictions[..., i:i+1]  # [batch_size, time_steps, 1]
            errors = targets - pred_q
            
            # Quantile loss
            loss_q = torch.where(
                errors >= 0,
                q * errors,
                (q - 1) * errors
            )
            losses.append(loss_q.mean())
        
        return torch.stack(losses).mean()


# Example usage function
def create_tft_model_example():
    """
    Example of how to create and use the TFT model.
    """
    # Model configuration
    model = TemporalFusionTransformer(
        hidden_layer_size=160,
        num_attention_heads=4,
        num_lstm_layers=1,
        num_attention_layers=1,
        dropout_rate=0.1,
        
        # Input specification (example with mixed inputs)
        num_static_categorical=2,
        num_static_continuous=3,
        num_historical_categorical=1,
        num_historical_continuous=5,
        num_future_categorical=1,
        num_future_continuous=2,
        
        # Time dimensions
        historical_steps=168,  # 7 days of hourly data
        prediction_steps=24,   # Predict next 24 hours
        
        # Output
        num_outputs=3,  # Predict 10th, 50th, 90th percentiles
        
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example forward pass
    batch_size = 32
    device = model.device
    
    # Create dummy inputs
    static_categorical = [
        torch.randint(0, 10, (batch_size,)).to(device),
        torch.randint(0, 5, (batch_size,)).to(device)
    ]
    static_continuous = [
        torch.randn(batch_size, 1).to(device) for _ in range(3)
    ]
    historical_categorical = [
        torch.randint(0, 7, (batch_size, 168)).to(device)
    ]
    historical_continuous = [
        torch.randn(batch_size, 168).to(device) for _ in range(5)
    ]
    future_categorical = [
        torch.randint(0, 7, (batch_size, 24)).to(device)
    ]
    future_continuous = [
        torch.randn(batch_size, 24).to(device) for _ in range(2)
    ]
    
    # Forward pass
    outputs = model(
        static_categorical=static_categorical,
        static_continuous=static_continuous,
        historical_categorical=historical_categorical,
        historical_continuous=historical_continuous,
        future_categorical=future_categorical,
        future_continuous=future_continuous
    )
    
    print(f"Predictions shape: {outputs['predictions'].shape}")
    # Expected: [32, 24, 3] - batch_size x prediction_steps x num_quantiles
    
    return model, outputs


# In[ ]:


class TFTEncoderOnly(nn.Module):
    """
    Encoder-only Temporal Fusion Transformer for historical pattern learning.
    
    This simplified version of TFT is designed for tasks where:
    - Only historical information is available (no future/known variables)
    - Output is a single prediction or set of predictions (not a time series)
    - Examples: predicting next period return, classification, risk scoring
    
    Architecture flow:
    1. Static and historical variable selection
    2. Static covariate encoding for context
    3. LSTM encoder for temporal processing
    4. Static enrichment of temporal features  
    5. Self-attention on historical sequence
    6. Temporal aggregation (pooling/attention-weighted)
    7. Output generation through feed-forward layers
    """
    
    def __init__(
        self,
        # Model architecture parameters
        hidden_layer_size: int = 160,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 1,
        num_attention_layers: int = 1,
        dropout_rate: float = 0.1,
        
        # Input specification
        num_static_categorical: int = 0,
        num_static_continuous: int = 0,
        num_historical_categorical: int = 0,
        num_historical_continuous: int = 0,
        
        # Embedding dimensions for categorical variables
        categorical_embedding_dims: Optional[Dict[str, int]] = None,
        
        # Time dimensions
        historical_steps: int = 30,  # Number of historical time steps (e.g., 30 candles)
        
        # Output specification
        output_size: int = 1,  # Number of outputs (e.g., 1 for single return prediction)
        output_type: str = 'regression',  # 'regression' or 'classification'
        num_classes: int = None,  # For classification tasks
        
        # Aggregation method for temporal features
        temporal_aggregation: str = 'attention',  # 'mean', 'last', 'attention', 'max'
        
        # Device
        device: str = 'cpu'
    ):
        """
        Initialize the Encoder-only TFT model.
        
        Args:
            hidden_layer_size: Size of hidden layers
            num_attention_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            num_attention_layers: Number of self-attention layers
            dropout_rate: Dropout rate
            num_static_categorical: Number of static categorical variables
            num_static_continuous: Number of static continuous variables
            num_historical_categorical: Number of historical categorical variables
            num_historical_continuous: Number of historical continuous variables
            categorical_embedding_dims: Embedding dimensions for categoricals
            historical_steps: Number of historical time steps
            output_size: Size of output vector
            output_type: Type of prediction task
            num_classes: Number of classes for classification
            temporal_aggregation: Method to aggregate temporal features
            device: Device to run on
        """
        super().__init__()
        
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_attention_heads = num_attention_heads
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_layers = num_attention_layers
        self.dropout_rate = dropout_rate
        
        self.num_static_categorical = num_static_categorical
        self.num_static_continuous = num_static_continuous
        self.num_historical_categorical = num_historical_categorical
        self.num_historical_continuous = num_historical_continuous
        
        self.historical_steps = historical_steps
        self.output_size = output_size
        self.output_type = output_type
        self.num_classes = num_classes if output_type == 'classification' else None
        self.temporal_aggregation = temporal_aggregation
        
        # Initialize embedding layers for categorical variables
        self.categorical_embeddings = nn.ModuleDict()
        if categorical_embedding_dims:
            for var_name, (vocab_size, embed_dim) in categorical_embedding_dims.items():
                self.categorical_embeddings[var_name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embed_dim
                ).to(device)
        
        # Input transformation layers
        self.static_continuous_transforms = nn.ModuleList([
            nn.Linear(1, hidden_layer_size).to(device) 
            for _ in range(num_static_continuous)
        ])
        
        self.historical_continuous_transforms = nn.ModuleList([
            nn.Linear(1, hidden_layer_size).to(device) 
            for _ in range(num_historical_continuous)
        ])
        
        # Variable Selection Networks
        if num_static_categorical + num_static_continuous > 0:
            self.static_variable_selection = VariableSelectionStatic(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                device=device
            )
            # eagerly build so params are visible to the optimizer
            self.static_variable_selection.build_layers(num_vars=num_static_categorical + num_static_continuous)
        else:
            self.static_variable_selection = None
            
        # Historical variable selection
        if num_historical_categorical + num_historical_continuous > 0:
            self.historical_variable_selection = VariableSelectionTemporal(
                hidden_layer_size=hidden_layer_size,
                static_context=True,
                dropout_rate=dropout_rate,
                device=device
            )
            self.historical_variable_selection.build_layers(num_vars=num_historical_categorical + num_historical_continuous)
        else:
            self.historical_variable_selection = None
        
        # Static Context Networks (for LSTM initialization and enrichment)
        if self.static_variable_selection is not None:
            self.static_context_module = StaticContexts(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                device=device
            )
        else:
            self.static_context_module = None
        
        # LSTM Encoder (only encoder, no decoder needed)
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        ).to(device)
        
        # Post-LSTM GRN for processing
        self.post_lstm_grn = TFTGRNLayer(
            device=device,
            hidden_layer_size=hidden_layer_size,
            output_size=None,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            additional_context=False,
            return_gate=False
        )
        
        # Static Enrichment Layer (optional, for enriching temporal features)
        if self.static_variable_selection is not None:
            self.static_enrichment = StaticEnrichmentLayer(
                hidden_layer_size=hidden_layer_size,
                context=True,
                dropout_rate=dropout_rate,
                device=device
            )
        else:
            self.static_enrichment = None
        
        # Self-Attention Stack (applied on historical sequence only)
        self.attention_stack = AttentionStack(
            num_layers=num_attention_layers,
            hidden_layer_size=hidden_layer_size,
            n_head=num_attention_heads,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Temporal Aggregation layers
        if temporal_aggregation == 'attention':
            # Learnable attention weights for aggregation
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.Tanh(),
                nn.Linear(hidden_layer_size, 1),
                nn.Softmax(dim=1)
            ).to(device)
        else:
            self.temporal_attention = None
        
        # Final processing before output
        self.final_grn = TFTGRNLayer(
            device=device,
            hidden_layer_size=hidden_layer_size,
            output_size=None, # FIXED: 26/09
            dropout_rate=dropout_rate,
            use_time_distributed=False,  # Applied on aggregated features
            additional_context=False,
            return_gate=False
        )
        
        # Output layers
        if output_type == 'regression':
            # For regression: map to output_size
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_layer_size, output_size)
            ).to(device)
        elif output_type == 'classification':
            # For classification: map to num_classes
            assert num_classes is not None, "num_classes must be specified for classification"
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_layer_size, num_classes)
            ).to(device)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        self.to(device)
    
    def _embed_categorical_buggy(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        """Embed categorical variable."""
        if var_name in self.categorical_embeddings:
            return self.categorical_embeddings[var_name](x)
        else:
            # Default: simple embedding
            embed_layer = nn.Embedding(
                x.max().item() + 1, 
                self.hidden_layer_size
            ).to(self.device)
            return embed_layer(x)
        
    def _embed_categorical(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        """
        Safely embed categorical variable.

        Args:
            x: Input tensor with categorical indices (may contain -1 for unknown)
            var_name: Name of the categorical variable

        Returns:
            Embedded tensor
        """
        if var_name in self.categorical_embeddings:
            embed_layer = self.categorical_embeddings[var_name]
            vocab_size = embed_layer.num_embeddings

            # Create safe indices
            x_safe = x.clone()

            # Map special values:
            # -1 (unknown) -> vocab_size - 1 (last index)
            # Any value >= vocab_size -> vocab_size - 1
            mask_unknown = (x_safe == -1)
            mask_overflow = (x_safe >= vocab_size)

            x_safe[mask_unknown] = vocab_size - 1
            x_safe[mask_overflow] = vocab_size - 1

            # Ensure all values are in valid range
            x_safe = torch.clamp(x_safe, min=0, max=vocab_size - 1)

            return embed_layer(x_safe)
        else:
            # Handle missing embedding gracefully
            # Create a learnable embedding on the fly
            max_idx = x.max().item() if x.numel() > 0 else 1
            vocab_size = max(max_idx + 2, 10)  # At least 10 for stability

            embed_layer = nn.Embedding(
                vocab_size, 
                self.hidden_layer_size,
                padding_idx=vocab_size - 1
            ).to(self.device)

            # Add to embeddings dict for future use
            self.categorical_embeddings[var_name] = embed_layer

            # Recursively call with the new embedding
            return self._embed_categorical(x, var_name)
    
    def _prepare_static_inputs(self, 
                              static_categorical: List[torch.Tensor], 
                              static_continuous: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare static inputs."""
        static_inputs = []
        
        # Process categorical variables
        for i, cat_var in enumerate(static_categorical):
            embedded = self._embed_categorical(cat_var.squeeze(), f"static_cat_{i}")
            static_inputs.append(embedded)
        
        # Process continuous variables  
        for i, cont_var in enumerate(static_continuous):
            if cont_var.dim() == 1:
                cont_var = cont_var.unsqueeze(-1)
            transformed = self.static_continuous_transforms[i](cont_var)
            static_inputs.append(transformed)
        
        return static_inputs
    
    def _prepare_historical_inputs(self,
                                  categorical_vars: List[torch.Tensor],
                                  continuous_vars: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare historical temporal inputs."""
        temporal_inputs = []
        
        # Process categorical variables
        for i, cat_var in enumerate(categorical_vars):
            # cat_var shape: [batch_size, time_steps]
            batch_size, time_steps = cat_var.shape
            
            # Reshape for embedding: [batch_size * time_steps]
            cat_var_flat = cat_var.reshape(-1)
            embedded = self._embed_categorical(cat_var_flat, f"historical_cat_{i}")
            
            # Reshape back: [batch_size, time_steps, embed_dim]
            embed_dim = embedded.shape[-1]
            embedded = embedded.reshape(batch_size, time_steps, embed_dim)
            temporal_inputs.append(embedded)
        
        # Process continuous variables
        for i, cont_var in enumerate(continuous_vars):
            # cont_var shape: [batch_size, time_steps] or [batch_size, time_steps, 1]
            if cont_var.dim() == 2:
                cont_var = cont_var.unsqueeze(-1)
            
            # Apply time-distributed transformation
            transformed = apply_time_distributed(
                self.historical_continuous_transforms[i], 
                cont_var
            )
            temporal_inputs.append(transformed)
        
        return temporal_inputs
    
    def aggregate_temporal_features(self, 
                                   temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features into a single vector.
        
        Args:
            temporal_features: [batch_size, time_steps, hidden_size]
            
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        if self.temporal_aggregation == 'mean':
            return temporal_features.mean(dim=1)
        
        elif self.temporal_aggregation == 'max':
            return temporal_features.max(dim=1)[0]
        
        elif self.temporal_aggregation == 'last':
            return temporal_features[:, -1, :]
        
        elif self.temporal_aggregation == 'attention':
            # Use attention weights to aggregate
            # attention_weights: [batch_size, time_steps, 1]
            attention_weights = self.temporal_attention(temporal_features)
            
            # Weighted sum: [batch_size, hidden_size]
            aggregated = (temporal_features * attention_weights).sum(dim=1)
            return aggregated
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.temporal_aggregation}")
    
    def forward(self,
                historical_continuous: Optional[List[torch.Tensor]] = None,
                historical_categorical: Optional[List[torch.Tensor]] = None,
                static_continuous: Optional[List[torch.Tensor]] = None,
                static_categorical: Optional[List[torch.Tensor]] = None,
                padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder-only TFT.
        
        Args:
            historical_continuous: List of continuous historical variables
                Each tensor shape: [batch_size, historical_steps]
            historical_categorical: List of categorical historical variables
                Each tensor shape: [batch_size, historical_steps]
            static_continuous: List of static continuous variables
                Each tensor shape: [batch_size, 1] or [batch_size]
            static_categorical: List of static categorical variables
                Each tensor shape: [batch_size, 1] or [batch_size]
            padding_mask: To help ignore left padded timeperiods
            
        Returns:
            Dictionary containing:
                - 'output': Final predictions [batch_size, output_size] or [batch_size, num_classes]
                - 'temporal_features': Processed temporal features
                - 'attention_output': Output from attention layers
                - 'historical_weights': Variable selection weights for historical inputs
                - 'static_weights': Variable selection weights for static inputs
        """
        outputs = {}
        
        # Get batch size
        batch_size = None
        for inputs in [historical_continuous, historical_categorical,
                      static_continuous, static_categorical]:
            if inputs is not None and len(inputs) > 0:
                batch_size = inputs[0].shape[0]
                break
        
        if batch_size is None:
            raise ValueError("No inputs provided")
        
        # 1. Process static inputs if available
        static_vec = None
        enrich_vec = None
        h0 = None
        c0 = None
        
        if self.static_variable_selection is not None and \
           (static_categorical or static_continuous):
            static_inputs = self._prepare_static_inputs(
                static_categorical or [],
                static_continuous or []
            )
            static_vec, static_weights = self.static_variable_selection(static_inputs)
            outputs['static_weights'] = static_weights
            
            # Generate context vectors
            stat_vec, enrich_vec, h0, c0 = self.static_context_module(static_vec)
        else:
            # No static inputs - use zeros for initialization
            stat_vec = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
            h0 = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
            c0 = torch.zeros(batch_size, self.hidden_layer_size).to(self.device)
        
        # 2. Process historical inputs with variable selection
        if self.historical_variable_selection is not None:
            historical_inputs = self._prepare_historical_inputs(
                historical_categorical or [],
                historical_continuous or []
            )
            
            if self.static_variable_selection is not None:
                # Use static context for variable selection
                historical_features, historical_weights = self.historical_variable_selection(
                    (historical_inputs, stat_vec)
                )
            else:
                # No static context
                historical_features, historical_weights = self.historical_variable_selection(
                    historical_inputs
                )
            outputs['historical_weights'] = historical_weights
        else:
            # No variable selection - create zero features
            historical_features = torch.zeros(
                batch_size, self.historical_steps, self.hidden_layer_size
            ).to(self.device)
        
        # 3. LSTM encoding of historical features
        # Initialize hidden states
        h0 = h0.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)  # [num_layers, batch, hidden]
        c0 = c0.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        
        lstm_output, (hn, cn) = self.lstm_encoder(historical_features, (h0, c0))
        
        # Apply post-LSTM GRN
        temporal_features = self.post_lstm_grn(lstm_output)
        outputs['temporal_features'] = temporal_features
        
        # 4. Static enrichment (if static features available)
        if self.static_enrichment is not None and enrich_vec is not None:
            enriched_features = self.static_enrichment((temporal_features, enrich_vec))
        else:
            enriched_features = temporal_features
        
        # 5. Self-attention on historical sequence
        # No causal mask needed since we're only looking at historical data
        # Use padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
        else:
            padding_mask = None
    
        attention_output = self.attention_stack(
            enriched_features,
            attn_mask=None,  # No causal mask for encoder-only
            padding_mask=padding_mask
        )
        outputs['attention_output'] = attention_output
        
        # 6. Aggregate temporal features into single vector
        aggregated_features = self.aggregate_temporal_features(attention_output)
        
        # 7. Final processing with GRN
        final_features = self.final_grn(aggregated_features)
        
        # 8. Generate output predictions
        output = self.output_layer(final_features)
        outputs['output'] = output
        
        # Add logits/probabilities for classification
        if self.output_type == 'classification':
            outputs['logits'] = output
            outputs['probabilities'] = F.softmax(output, dim=-1)
            outputs['predictions'] = torch.argmax(output, dim=-1)
        
        return outputs
    
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        Convenience method that returns only the predictions.
        
        Returns:
            For regression: predictions [batch_size, output_size]
            For classification: predicted classes [batch_size]
        """
        outputs = self.forward(*args, **kwargs)
        
        if self.output_type == 'classification':
            return outputs['predictions']
        else:
            return outputs['output']
    
    def get_feature_importance(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get feature importance from variable selection networks.
        
        Returns:
            Dictionary with 'static_weights' and 'historical_weights'
        """
        outputs = self.forward(*args, **kwargs)
        importance = {}
        
        if 'static_weights' in outputs:
            importance['static'] = outputs['static_weights']
        if 'historical_weights' in outputs:
            importance['historical'] = outputs['historical_weights']
            
        return importance


# Example usage for stock return prediction
def create_stock_prediction_example():
    """
    Example: Predict average return over next 3 candles using past 30 candles.
    """
    
    # Model configuration for stock prediction
    model = TFTEncoderOnly(
        hidden_layer_size=128,
        num_attention_heads=4,
        num_lstm_layers=2,
        num_attention_layers=1,
        dropout_rate=0.1,
        
        # Static features (e.g., stock ticker, sector, market cap category)
        num_static_categorical=2,  # ticker_id, sector
        num_static_continuous=1,    # log_market_cap
        
        # Historical features (OHLCV + technical indicators)
        num_historical_categorical=2,  # day_of_week, is_market_open
        num_historical_continuous=10,   # open, high, low, close, volume, RSI, MA_20, etc.
        
        # Time configuration
        historical_steps=30,  # Past 30 candles
        
        # Output configuration
        output_size=1,  # Single value: average return over next 3 candles
        output_type='regression',
        
        # Aggregation
        temporal_aggregation='attention',  # Learn which historical steps are important
        
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy data
    batch_size = 32
    device = model.device
    
    # Historical continuous features (OHLCV data normalized)
    historical_continuous = [
        torch.randn(batch_size, 30).to(device),  # open
        torch.randn(batch_size, 30).to(device),  # high  
        torch.randn(batch_size, 30).to(device),  # low
        torch.randn(batch_size, 30).to(device),  # close
        torch.randn(batch_size, 30).to(device),  # volume
        torch.randn(batch_size, 30).to(device),  # RSI
        torch.randn(batch_size, 30).to(device),  # MA_20
        torch.randn(batch_size, 30).to(device),  # Bollinger_upper
        torch.randn(batch_size, 30).to(device),  # Bollinger_lower
        torch.randn(batch_size, 30).to(device),  # MACD
    ]
    
    # Historical categorical features
    historical_categorical = [
        torch.randint(0, 7, (batch_size, 30)).to(device),    # day_of_week
        torch.randint(0, 2, (batch_size, 30)).to(device),    # is_market_open
    ]
    
    # Static features
    static_categorical = [
        torch.randint(0, 500, (batch_size,)).to(device),  # ticker_id (500 stocks)
        torch.randint(0, 11, (batch_size,)).to(device),   # sector (11 sectors)
    ]
    static_continuous = [
        torch.randn(batch_size, 1).to(device),  # log_market_cap
    ]
    
    # Forward pass
    outputs = model(
        historical_continuous=historical_continuous,
        historical_categorical=historical_categorical,
        static_continuous=static_continuous,
        static_categorical=static_categorical
    )
    
    print(f"Predictions shape: {outputs['output'].shape}")  # [32, 1]
    print(f"Historical weights shape: {outputs['historical_weights'].shape}")  # [32, 30, 12]
    
    # Training example
    target_returns = torch.randn(batch_size, 1).to(device)  # Actual returns
    
    # MSE loss for regression
    loss = F.mse_loss(outputs['output'], target_returns)
    print(f"Loss: {loss.item():.4f}")
    
    return model, outputs


# Example for multi-class classification
def create_classification_example():
    """
    Example: Classify market regime (trending up/down/sideways) from historical data.
    """
    
    model = TFTEncoderOnly(
        hidden_layer_size=128,
        num_attention_heads=4,
        num_lstm_layers=1,
        num_attention_layers=1,
        dropout_rate=0.2,
        
        # Only historical features, no static
        num_static_categorical=0,
        num_static_continuous=0,
        num_historical_categorical=1,
        num_historical_continuous=5,
        
        historical_steps=50,
        
        # Classification setup
        output_type='classification',
        num_classes=3,  # Up, Down, Sideways
        
        temporal_aggregation='last',  # Use last hidden state
        
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    batch_size = 16
    device = model.device
    
    # Historical data
    historical_continuous = [
        torch.randn(batch_size, 50).to(device) for _ in range(5)
    ]
    historical_categorical = [
        torch.randint(0, 4, (batch_size, 50)).to(device)  # Quarter of year
    ]
    
    outputs = model(
        historical_continuous=historical_continuous,
        historical_categorical=historical_categorical
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")  # [16, 3]
    print(f"Predictions: {outputs['predictions']}")     # Class indices
    print(f"Probabilities: {outputs['probabilities'].shape}")  # [16, 3]
    
    # Training with cross-entropy
    target_classes = torch.randint(0, 3, (batch_size,)).to(device)
    loss = F.cross_entropy(outputs['logits'], target_classes)
    print(f"Classification loss: {loss.item():.4f}")
    
    return model, outputs

