from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from rtdl.modules import _INTERNAL_ERROR_MESSAGE
from torch import Tensor

from .core import FeatureTokenizer, FTTransformer, Head, Transformer


class FTTransformerWithMaskToken(FTTransformer):
    def __init__(
        self,
        num_features,
        feature_tokenizer: FeatureTokenizer,
        transformer: Transformer,
        head: Head,
    ) -> None:
        super().__init__(num_features, feature_tokenizer, transformer, head)
        self.mask_token = nn.Parameter(Tensor(1, 1, feature_tokenizer.d_token))
        self.initialization_.apply(self.mask_token, feature_tokenizer.d_token)

    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """The replacement for :code:`.parameters()` when creating optimizers.

        Example::

            optimizer = AdamW(
                model.optimization_param_groups(), lr=1e-4, weight_decay=1e-5
            )
        """
        no_wd_names = ["feature_tokenizer", "normalization", ".bias", "pos_embedding", "mask_token"]
        assert isinstance(getattr(self, no_wd_names[0], None), FeatureTokenizer), _INTERNAL_ERROR_MESSAGE
        assert (
            sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name)
            == len(self.transformer.blocks) * 2
            - int("attention_normalization" not in self.transformer.blocks[0])  # type: ignore
            + 1
        ), _INTERNAL_ERROR_MESSAGE

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        return [
            {"params": [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                "params": [v for k, v in self.named_parameters() if not needs_wd(k)],
                "weight_decay": 0.0,
            },
        ]

    def random_masking(self, x: Tensor, mask_ratio: float) -> Tensor:
        """
        Args:
            x (tensor): the input of shape (b, n, d).
            mask_ratio (float): the ratio of data points to be masked

        Returns:
            x_masked (tensor): the output of shape (b, n, d) representing the masked data
        """
        b, n, d = x.shape
        mask = torch.bernoulli(torch.ones(b, n) * mask_ratio)
        mask = mask.unsqueeze(-1).to(torch.float).to(x.device)
        mask = mask.repeat(1, 1, d)

        mask_tokens = self.mask_token.repeat(b, n, 1)
        x_masked = x * (1 - mask) + mask_tokens * mask
        return x_masked

    def forward(
        self,
        x_num: Optional[Tensor],
        x_cat: Optional[Tensor],
        mask_ratio: float = 0.0,
        bias_after_mask: bool = True,
    ) -> Tensor:
        """
        Args:
            x_num (tensor): the input of shape (b, n) representing the numerical values
            x_cat (tensor): the input of shape (b, n) representing the categorical values
            mask_ratio (float): the ratio of data points to be masked
            bias_after_mask (bool): whether to add the positional embedding before or after masking
        Returns:
            tensor: the output of shape (b, num_classes) representing the model's prediction
        """
        x = self.feature_tokenizer(x_num, x_cat)

        if not bias_after_mask:
            x = x + self.pos_embedding
        if mask_ratio > 0.0:
            x = self.random_masking(x, mask_ratio)
        if bias_after_mask:
            x = x + self.pos_embedding

        x = self.cls_token(x)
        x = self.transformer(x)
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        return self.head(x)
