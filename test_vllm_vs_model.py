import vllm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from examples.run_vlm_grpo import hf_data_processor, resolve_to_image
from nemo_rl.data.hf_datasets.clevr import CLEVRCoGenTDataset, format_clevr_cogent_dataset 

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    llm = vllm.LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=4, enforce_eager=True)
    # model_hf = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True)

    # dataset
    dataset = CLEVRCoGenTDataset(split="trainA", seed=42, task_name="clevr-cogent", prompt_file="examples/prompts/clevr_cogent_cot.txt")
    dataset = dataset.formatted_ds["train"]

    idx = 0
    datum = format_clevr_cogent_dataset(dataset[idx])
    # get vllm log
    vllm_message = {
        'prompt': processor.apply_chat_template(datum['messages'][:1], tokenize=False, add_generation_prompt=True),
        'multi_modal_data': {
            'image': [resolve_to_image(datum['messages'][0]['content'][0]['image'])],
        }
    }

    # processor model
    processor_input = processor.apply_chat_template(datum['messages'][:1], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = processor_input['input_ids'][0]

    breakpoint()

    # get vllm log
    llm_output = llm.generate([vllm_message])[0]
    input_tokens = llm_output.outputs[0].prompt_token_ids
    out_tokens = llm_output.outputs[0].token_ids

    breakpoint()
