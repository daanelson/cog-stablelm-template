from argparse import ArgumentParser
import os
import stat
from jinja2 import Template

# llama configs, can modify as needed. 
TRAIN_DEPENDENCIES = ["deepspeed==0.8.3","git+https://github.com/huggingface/peft.git@632997d1fb776c3cf05d8c2537ac9a98a7ce9435"]


CONFIGS = {
    "stablelm-base-alpha-3b": {
        "cog_yaml_parameters": {"fine_tune":'''train: "train.py:train"''', "extra_deps": TRAIN_DEPENDENCIES},
        "config_py_parameters": {"model_name": "StabilityAI/stablelm-base-alpha-3b", "tensorizer_weights": "gs://replicate-weights/stablelm-base-alpha-3b-fp16.tensors", "instruction_tuned": "False"}
    },
    "stablelm-base-alpha-7b": {
        "cog_yaml_parameters": {"fine_tune":'''train: "train.py:train"''', "extra_deps": TRAIN_DEPENDENCIES},
        "config_py_parameters": {"model_name": "StabilityAI/stablelm-base-alpha-7b", "tensorizer_weights": "gs://replicate-weights/stablelm-base-alpha-7b-fp16.tensors", "instruction_tuned": "False"}
    },
    "stablelm-tuned-alpha-7b": {
        "cog_yaml_parameters": {"fine_tune":"", "extra_deps": []},
        "config_py_parameters": {"model_name": "StabilityAI/stablelm-tuned-alpha-7b", "tensorizer_weights": "gs://replicate-weights/stablelm-tuned-alpha-7b.tensors", "instruction_tuned": "True"}
    },

}

def _reset_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_one_config(template_fpath: str, fname_out: str, config: dict):
    with open(template_fpath, "r") as f:
        template_content = f.read()
        base_template = Template(template_content)

    _reset_file(fname_out)

    with open(fname_out, "w") as f:
        f.write(base_template.render(config))

    # Give all users write access to resulting generated file. 
    current_permissions = os.stat(fname_out).st_mode
    new_permissions = current_permissions | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    os.chmod(fname_out, new_permissions)


def write_configs(model_name):
    master_config = CONFIGS[model_name]
    write_one_config("templates/cog_template.yaml", "cog.yaml", master_config['cog_yaml_parameters'])
    write_one_config("templates/config_template.py", "config.py", master_config['config_py_parameters'])

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="stablelm-base-alpha-3b", help="name of the flan-t5 model you want to configure cog for")
    args = parser.parse_args()

    write_configs(args.model_name)