import os
from typing import List, Optional, Union, Dict
from transformers import BertTokenizer

class ZeusTokenizer(BertTokenizer):
    pretrained_resource_files_map = {}

    def __init__(self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        bos_token="[START]",
        eos_token="[gEND]",
        mask_token="[gMASK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
        padding_side="right",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs):
        self.name = "ZeusTokenizer"

        super().__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            padding_side=padding_side,
            **kwargs,
        )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
            在特殊标记中构建输入，包括第一个和第二个输入。如果存在第二个输入，则忽略。
        
        Args:
            token_ids_0 (List[int]): 第一个输入的标记ID列表。
            token_ids_1 (Optional[List[int]], optional): 第二个输入的标记ID列表（默认为None）。
        
        Returns:
            List[int]: 包含第一个和第二个输入以及特殊标记的标记ID列表。
        
        Raises:
            None
        """
        if token_ids_1 is not None:
            logger.warning("Support single input text and the second one is ignored.")
        return token_ids_0 + [self.mask_token_id]
