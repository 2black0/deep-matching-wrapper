import os
import torch
from yacs.config import CfgNode as CN

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        print(f"[{level}] {message}")


def detect_NaN(feat_0, feat_1):
    print(f"NaN detected in feature")
    print(
        f"#NaN in feat_0: {torch.isnan(feat_0).int().sum()}, #NaN in feat_1: {torch.isnan(feat_1).int().sum()}"
    )
    feat_0[torch.isnan(feat_0)] = 0
    feat_1[torch.isnan(feat_1)] = 0
