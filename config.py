import logging
import subprocess
import time
from collections import OrderedDict
import os

from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEFAULT_MODEL_NAME = "StabilityAI/stablelm-base-alpha-3b" #"StabilityAI/stablelm-base-alpha-3b"  # path from which we pull weights when there's no COG_WEIGHTS environment variable, + config
TENSORIZER_WEIGHTS_PATH = "gs://replicate-weights/stablelm-base-alpha-3b-fp16.tensors"
INSTRUCTION_TUNED = False
LOCAL_PATH = f'''/src/{DEFAULT_MODEL_NAME.split("/")[-1].replace("-", "_")}.tensors'''

TOKENIZER_PATH = "/src/tokenizer"
CONFIG_LOCATION = DEFAULT_MODEL_NAME
CACHE_DIR = "pretrained_weights"


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


def format_prompt(prompt):
    if INSTRUCTION_TUNED:
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
        return f"{system_prompt}<|USER|>{prompt}<|ASSISTANT|>"
    return prompt


def maybe_download(path=TENSORIZER_WEIGHTS_PATH):
    st = time.time()    
    output_path = LOCAL_PATH
    print(f"Downloading tensors to {output_path}")
    if path.startswith("gs://") and not os.path.exists(output_path):
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        print(f"Tensors downloaded in {time.time() - st}")
        return output_path
    elif os.path.exists(output_path):
        return output_path
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
    except Exception as e:
        print(f"Exception loading tensorized weights: {e}")
        print(f"Loading weights via hf")
        model = load_huggingface_model(cls)
        return model


def load_huggingface_model(cls):
    st = time.time()
    print(f"loading weights w/o tensorizer")

    model = cls.from_pretrained(DEFAULT_MODEL_NAME, cache_dir=CACHE_DIR).to("cuda:0")
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
