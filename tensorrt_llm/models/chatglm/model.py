# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import Tensor, concat, shape, gelu
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm,
                       RmsNorm)
from ...module import Module, ModuleList
from ..modeling_utils import DecoderModelForCausalLM, PretrainedConfig


class ChatGLMDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.chatglm_version = config.chatglm_version

        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        layernorm_epsilon = config.norm_epsilon

        rope_base = 10000.0 * config.rope_ratio
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.alpha = (2 * config.num_hidden_layers)**0.5
        norm_cls = RmsNorm if config.rmsnorm else LayerNorm

        if config.chatglm_version == 'glm':
            attention_mask_type = AttentionMaskType.causal
        elif config.chatglm_version == 'chatglm':
            attention_mask_type = AttentionMaskType.bidirectional
        elif config.chatglm_version in ['chatglm2', 'chatglm3']:
            attention_mask_type = AttentionMaskType.causal

        self.input_layernorm = norm_cls(
            normalized_shape=hidden_size,
            eps=layernorm_epsilon,
            elementwise_affine=True,
            dtype=dtype,
        )

        self.attention = Attention(
            hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
            attention_mask_type=attention_mask_type,
            bias=config.add_qkv_bias,
            dense_bias=config.add_bias_linear,
            dtype=config.dtype,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=rope_base,
            rotary_embedding_scaling=None,
            rotary_embedding_percentage=0.5,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=config.quant_mode,
            q_scaling=1.0,
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
        )

        mlp_hidden_size = hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=config.hidden_act,
            bias=config.add_bias_linear,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
        )

        self.post_layernorm = norm_cls(
            normalized_shape=hidden_size,
            eps=layernorm_epsilon,
            elementwise_affine=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        use_cache: bool = False,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):
        norm_output = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states=norm_output,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            encoder_output=None,
            position_embedding=position_ids,
        )

        if use_cache:
            attention_output, presents = attention_output

        if self.chatglm_version == 'chatglm':
            residual = norm_output

            norm_input = residual * self.alpha + attention_output

            norm_output = self.post_layernorm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output

            output = residual * self.alpha + mlp_output

        else:
            residual = norm_output if self.apply_residual_connection_post_layernorm else hidden_states

            norm_input = residual + attention_output

            norm_output = self.post_layernorm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output if self.apply_residual_connection_post_layernorm else norm_input

            output = residual + mlp_output

        if use_cache:
            return (output, presents)
        return output


class ChatGLMModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.chatglm_version = config.chatglm_version
        norm_cls = RmsNorm if config.rmsnorm else LayerNorm

        if config.use_parallel_embedding:
            self.vocab_embedding = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                sharding_dim=config.embedding_sharding_dim,
                tp_rank=config.mapping.tp_rank)
        else:
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        if config.chatglm_version == 'glm':
            self.position_embedding = Embedding(
                config.max_position_embeddings + 1,
                config.hidden_size,
                dtype=config.dtype,
            )
            self.block_embedding = Embedding(
                config.max_position_embeddings + 1,
                config.hidden_size,
                dtype=config.dtype,
            )

        self.layers = ModuleList([
            ChatGLMDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ])

        self.ln_f = norm_cls(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            elementwise_affine=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        use_cache: bool = False,
        attention_mask: Tensor = None,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):
        hidden_states = self.vocab_embedding(input_ids)

        if self.chatglm_version == 'glm':
            if default_net().plugin_config.remove_input_padding:
                position_ids_list = position_ids.split(1, dim=0)
            else:
                position_ids_list = position_ids.split(1, dim=1)

            position_embedding = self.position_embedding(position_ids_list[0])
            block_embedding = self.block_embedding(position_ids_list[1])
            position_embedding = position_embedding + block_embedding

            if default_net().plugin_config.remove_input_padding:
                position_embedding = position_embedding.view(
                    concat([
                        shape(position_embedding, 1),
                        shape(position_embedding, 2)
                    ]))
            else:
                position_embedding = position_embedding.view(
                    concat([
                        shape(position_embedding, 0),
                        shape(position_embedding, 2),
                        shape(position_embedding, 3),
                    ]))

            hidden_states = hidden_states + position_embedding

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes):
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
            )

            if use_cache:
                hidden_states = layer_output[0]
                presents.append(layer_output[1])

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states

class ChatGLMLMHead(Module):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        
        self.fc = ColumnLinear(
            hidden_size,
            hidden_size,
            bias=True,
            dtype=dtype,
            tp_group=None,
            tp_size=1
        )
        self.layernorm = LayerNorm(
            normalized_shape=hidden_size,
            elementwise_affine=True,
            dtype=dtype
        )
        self.proj = ColumnLinear(
            hidden_size,
            vocab_size,
            bias=True,
            dtype=dtype,
            tp_group=None,
            tp_size=1
        )
        self.dtype = dtype

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        logits = self.proj(hidden_states)

        return logits

class ChatGLMForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = ChatGLMModel(config)
        vocab_size_padded = pad_vocab_size(config.output_vocab_size,
                                           config.mapping.tp_size)

        # share_weight = None
        # if config.share_embedding_table:
        #     share_weight = transformer.vocab_embedding.weight
        lm_head = ChatGLMLMHead(
            config.hidden_size,
            vocab_size_padded,
            dtype=config.dtype
        )
        super().__init__(config, transformer, lm_head)

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('chatglm_version', 'chatglm3')
        config.set_if_not_exist('add_bias_linear', False)
        config.set_if_not_exist('add_qkv_bias', True)
        config.set_if_not_exist('apply_query_key_layer_scaling', False)
        config.set_if_not_exist('apply_residual_connection_post_layernorm',
                                False)
        config.set_if_not_exist('rmsnorm', True)
        config.set_if_not_exist('rope_ratio', 1.0)

    def prepare_inputs(self, *args, **kwargs):
        """See `PretrainedModel.prepare_inputs` for the detailed parameter list.
        """
        if self.transformer.chatglm_version in ['chatglm', 'glm']:
            position_encoding_2d = True
        else:
            position_encoding_2d = False
        return super().prepare_inputs(*args,
                                      **kwargs,
                                      position_encoding_2d=position_encoding_2d)
