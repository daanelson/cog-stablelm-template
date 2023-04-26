import logging
import re
import subprocess
import time
from collections import OrderedDict

import torch
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEFAULT_MODEL_NAME = "StabilityAI/stablelm-base-alpha-3b"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
# DEFAULT_MODEL_NAME = "distilgpt2"
TOKENIZER_NAME = DEFAULT_MODEL_NAME
CONFIG_LOCATION = DEFAULT_MODEL_NAME


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir="pretrained_weights")
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


def load_model(model_name_or_path=DEFAULT_MODEL_NAME):
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir="pretrained_weights")


