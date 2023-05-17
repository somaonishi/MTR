from typing import Any, Dict, List, Optional, Type, Union

import rtdl
import torch
import torch.nn as nn
from rtdl.modules import (
    _INTERNAL_ERROR_MESSAGE,
    CategoricalFeatureTokenizer,
    NumericalFeatureTokenizer,
    _TokenInitialization,
)
from torch import Tensor

from .transformer import Transformer


class FeatureTokenizer(rtdl.modules.FeatureTokenizer):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
        bias: bool = False,
    ) -> None:
        super().__init__(n_num_features, cat_cardinalities, d_token)
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=bias,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(cat_cardinalities, d_token, bias, self.initialization)
            if cat_cardinalities
            else None
        )


class Head(nn.Module):
    """The final module of the `Transformer` that performs BERT-like inference."""

    def __init__(
        self,
        *,
        d_in: int,
        bias: bool,
        d_out: int,
    ):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, projection_dim=64) -> None:
        super().__init__()
        self.f1 = nn.Linear(d_in, d_in, bias=False)
        self.act = nn.ReLU()
        self.f2 = nn.Linear(d_in, projection_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f1(x)
        x = self.act(x)
        x = self.f2(x)
        return x


class FTTransformer(rtdl.FTTransformer):
    def __init__(
        self,
        num_features,
        feature_tokenizer: FeatureTokenizer,
        transformer: Transformer,
        head: Head,
    ) -> None:
        super().__init__(feature_tokenizer, transformer)
        self.pos_embedding = nn.Parameter(Tensor(num_features, feature_tokenizer.d_token))
        self.initialization_ = _TokenInitialization.from_str("uniform")
        self.initialization_.apply(self.pos_embedding, feature_tokenizer.d_token)

        if transformer.prenormalization:
            self.normalization = nn.LayerNorm(feature_tokenizer.d_token)
        else:
            self.normalization = nn.Identity()
        self.activation = nn.ReLU()
        self.head = head

    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """The replacement for :code:`.parameters()` when creating optimizers.

        Example::

            optimizer = AdamW(
                model.optimization_param_groups(), lr=1e-4, weight_decay=1e-5
            )
        """
        no_wd_names = ["feature_tokenizer", "normalization", ".bias", "pos_embedding"]
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

    def make_default_optimizer(self, lr=1e-4) -> torch.optim.AdamW:
        """Make the optimizer for the default FT-Transformer."""
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            lr=lr,
            weight_decay=1e-5,
        )

    @classmethod
    def make_default(
        cls: Type["FTTransformer"],
        *,
        num_features: int,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        n_blocks: int = 3,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        d_out: int,
    ) -> "FTTransformer":
        transformer_config = cls.get_default_transformer_config(n_blocks=n_blocks)
        for arg_name in [
            "last_layer_query_idx",
            "kv_compression_ratio",
            "kv_compression_sharing",
            "d_out",
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(num_features, n_num_features, cat_cardinalities, transformer_config)

    @classmethod
    def _make(
        cls,
        num_features,
        n_num_features,
        cat_cardinalities,
        transformer_config,
    ):
        feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=transformer_config["d_token"],
            bias=False,
        )
        if transformer_config["d_out"] is None:
            transformer_config["head_activation"] = None
        if transformer_config["kv_compression_ratio"] is not None:
            transformer_config["n_tokens"] = feature_tokenizer.n_tokens + 2

        head = Head(
            d_in=transformer_config["d_token"],
            d_out=transformer_config["d_out"],
            bias=True,
        )
        del (
            transformer_config["d_out"],
            transformer_config["head_activation"],
            transformer_config["head_normalization"],
        )
        return cls(
            num_features,
            feature_tokenizer,
            transformer=Transformer(**transformer_config),
            head=head,
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        x = self.feature_tokenizer(x_num, x_cat)
        x = x + self.pos_embedding
        x = self.cls_token(x)
        x = self.transformer(x)
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        return self.head(x)
