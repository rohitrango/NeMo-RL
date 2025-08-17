# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.eval_datasets import load_eval_dataset
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.vlm_environment import VLMEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config

from transformers import AutoProcessor

from collections import defaultdict
from nemo_rl.data.interfaces import TaskDataSpec, TaskDataProcessFnCallable
from examples.run_vlm_grpo import hf_data_processor

TokenizerType = PreTrainedTokenizerBase

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides

def load_vlm_dataset(data_config):
    name = data_config["dataset_name"]

    if name == "clevr-cogent":
        from nemo_rl.data.hf_datasets.clevr import CLEVRCoGenTDataset
        return CLEVRCoGenTDataset(
            split=data_config["split"],
            seed=data_config["seed"],
            task_name=data_config["task_name"],
        )
    else:
        raise ValueError(f"No processor for dataset {name}.")


def setup_data(processor: AutoProcessor, data_config, env_configs, config):
    print("Setting up data...")
    # load dataset
    base_dataset = load_vlm_dataset(data_config)

    if 'sft' in config["generation"]["model_name"]:
        print("SFT mode detected, setting prompt_file to None")
        data_config['prompt_file'] = None

    task_name = data_config["task_name"]
    vlm_task_spec = TaskDataSpec(
        task_name=data_config["task_name"],
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    env = VLMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vlm_environment.VLMEnvironment"
            )
        }
    ).remote(env_configs[data_config["task_name"]])

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (vlm_task_spec, hf_data_processor))
    )
    task_data_processors[task_name] = (vlm_task_spec, hf_data_processor)

    dataset = AllTaskProcessedDataset(
        dataset=base_dataset.formatted_ds["validation"],
        tokenizer=processor,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, processor


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "evals", "eval_vlm.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    print(f"Running evaluation on {config['data']['split']} split")
    # Setup data
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"], config)

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(
        vllm_generation,
        dataloader,
        env,
        master_config,
    )


if __name__ == "__main__":
    main()
