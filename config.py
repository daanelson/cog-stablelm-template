import logging
import re
import os
import subprocess
import time
from collections import OrderedDict
import logging

import torch
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEFAULT_MODEL_NAME = "StabilityAI/stablelm-base-alpha-3b"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
TENSORIZER_WEIGHTS_PATH = "gs://replicate-weights/stablelm-tuned-alpha-7b.tensors"
CACHE_DIR = "pretrained_weights"

SYSTEM_PROMPT = """"""

TOKENIZER_NAME = DEFAULT_MODEL_NAME
CONFIG_LOCATION = DEFAULT_MODEL_NAME


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=CACHE_DIR)
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


def maybe_download(path):
    st = time.time()
    print(f"Downloading tensors")
    output_path = "/tmp/weights.tensors" # TODO - templatize
    if path.startswith("gs://") and not os.path.exists(output_path):
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        return output_path
    print(f"Tensors downloaded in {time.time() - st}")
    return path


def load_model(plaid_mode=True, cls=AutoModelForCausalLM):
    try:
        print("Loading tensorized weights from public path")
        model = load_tensorizer(
            weights=maybe_download(TENSORIZER_WEIGHTS_PATH),
            plaid_mode=plaid_mode,
            cls=cls
        )
        return model
    except:
        print("Loading via hf")
        model = load_huggingface_model()
        return model


def load_huggingface_model():
    st = time.time()
    print(f"loading weights w/o tensorizer")

    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, cache_dir=CACHE_DIR).to("cuda:0")
    print(f"weights loaded in {time.time() - st}")
    return model


def load_tensorizer(weights, plaid_mode=True, cls=AutoModelForCausalLM):
    st = time.time()
    print(f"deserializing weights from {weights}")
    config = AutoConfig.from_pretrained(DEFAULT_MODEL_NAME, cache_dir=CACHE_DIR)
    
    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: cls.from_pretrained(
            None, config=config, state_dict=OrderedDict()
        )
    )
    logging.disable(logging.NOTSET)

    des = TensorDeserializer(weights, plaid_mode=plaid_mode)
    des.load_into_module(model)
    print(f"weights loaded in {time.time() - st}")
    return model

