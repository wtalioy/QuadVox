# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from ..transformer.activation import Swish
from ..transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from ..transformer.embedding import (PositionalEncoding,
                                             RelPositionalEncoding,
                                             WhisperPositionalEncoding,
                                             LearnablePositionalEncoding,
                                             NoPositionalEncoding)
from ..transformer.attention import (MultiHeadedAttention,
                                             RelPositionMultiHeadedAttention)
from ..transformer.embedding import EspnetRelPositionalEncoding
from ..transformer.subsampling import LegacyLinearNoSubsampling
from ..llm.llm import TransformerLM, Qwen2LM, CosyVoice3LM
from ..flow.flow import MaskedDiffWithXvec, CausalMaskedDiffWithXvec, CausalMaskedDiffWithDiT
from ..hifigan.generator import HiFTGenerator, CausalHiFTGenerator
from ..cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    # Use type name checking as fallback in case isinstance fails due to module path differences
    llm_type_name = type(configs['llm']).__name__
    flow_type_name = type(configs['flow']).__name__
    hift_type_name = type(configs['hift']).__name__
    
    if isinstance(configs['llm'], TransformerLM) and isinstance(configs['flow'], MaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoiceModel
    if isinstance(configs['llm'], Qwen2LM) and isinstance(configs['flow'], CausalMaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoice2Model
    if isinstance(configs['llm'], CosyVoice3LM) and isinstance(configs['flow'], CausalMaskedDiffWithDiT) and isinstance(configs['hift'], CausalHiFTGenerator):
        return CosyVoice3Model
    
    # Fallback: check by type name (for cases where isinstance fails due to module path differences)
    if llm_type_name == 'CosyVoice3LM' and flow_type_name == 'CausalMaskedDiffWithDiT' and hift_type_name == 'CausalHiFTGenerator':
        return CosyVoice3Model
    if llm_type_name == 'Qwen2LM' and flow_type_name == 'CausalMaskedDiffWithXvec' and hift_type_name == 'HiFTGenerator':
        return CosyVoice2Model
    if llm_type_name == 'TransformerLM' and flow_type_name == 'MaskedDiffWithXvec' and hift_type_name == 'HiFTGenerator':
        return CosyVoiceModel
    
    raise TypeError(f'No valid model type found! llm={llm_type_name}, flow={flow_type_name}, hift={hift_type_name}')
