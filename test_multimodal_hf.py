# script to test the parameters of multimodal models

from transformers import AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextVideoProcessor
from PIL import Image
import torch
import requests
import base64
from io import BytesIO
from collections import defaultdict

# autoprocessor factory for models 
processor_dict = defaultdict(lambda: AutoProcessor)
processor_dict["llava-hf/llava-v1.6-34b-hf"] = LlavaNextProcessor
processor_dict["llava-hf/llava-v1.6-mistral-7b-hf"] = LlavaNextProcessor
processor_dict["llava-hf/LLaVA-NeXT-Video-7B-hf-DPO"] = LlavaNextVideoProcessor

def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """ Resolve the image path to a PIL.Image object. 
    
    image_path can be either:
    - path to local file
    - url to image
    - base64 encoded image
    """
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        # Handle URL
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    elif image_path_or_image.startswith("data:"):
        # Handle base64 encoded image
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data))
    else:
        # Handle local file path
        return Image.open(image_path_or_image)

def downscale_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    '''
    Downscale the image such that largest dimension is at most max_size.
    dont do anything for small images.
    '''
    width, height = image.size  
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
            return image.resize((new_width, new_height))
        else:
            return image
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height))

def preprocess_conversation_phi4(conversation):
    # get new conversation with tokenizer
    new_conversation = []
    image_id = 1
    audio_id = 1
    for turn in conversation:
        new_turn = {}
        new_turn['role'] = turn['role']
        new_turn['content'] = ""
        # if just text, copy it, else create a text string with placeholders
        if isinstance(turn['content'], str):
            new_turn['content'] = turn['content']
        elif isinstance(turn['content'], list):
            for item in turn['content']:
                if item['type'] == 'text':
                    new_turn['content'] += item['text']
                elif item['type'] == 'image':
                    new_turn['content'] += f"<|image_{image_id}|>"
                    image_id += 1
                elif item['type'] == 'audio':
                    new_turn['content'] += f"<|audio_{audio_id}|>"
                    audio_id += 1
                else:
                    raise ValueError(f"Unexpected content type: {item['type']}")
        # else:
            # raise ValueError(f"Unexpected content type: {type(turn['content'])}")
        new_conversation.append(new_turn)
    return new_conversation

def summarize_conversation_process(processor, conversation, images_conversation, conversation_name):
    '''
    Summarize the conversation process.
    '''
    print('-'*100)
    print(f"Conversation Name: {conversation_name}")
    prompt_text = processor.apply_chat_template(
        # preprocess_conversation_phi4(conversation),
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors=False,
    )


    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]

    # for msg in conversation:
    #     prompt_text = processor.apply_chat_template(
    #         [msg],
    #         tokenize=False,
    #         return_tensors=False,
    #     )
    #     print(f"Prompt Text: ```{prompt_text}```")
    #     images = [resolve_to_image(content["image"]) for content in msg["content"] if isinstance(content, dict) and content["type"] == "image"]
    #     images = [downscale_image(image) for image in images]
    #     images = None if len(images) == 0 else images
    #     tensordict = processor(text=prompt_text, images=images, return_tensors="pt", add_special_tokens=True)
    #     for key, value in tensordict.items():
    #         print(f"{key}: {value.shape}")
    #     print('-'*100)

    # print("\n\n")

    print(f"Prompt Text: ```{prompt_text}```")
    
    # now apply the processor to the prompt text
    print(f"Images: {images_conversation}")
    tensordict = processor(text=prompt_text, images=images_conversation, return_tensors="pt") #, add_special_tokens=True) -- doesnt work for phi-4

    for key, value in tensordict.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    if 'token_type_ids' in tensordict:
        print(f"Token Type IDs: {tensordict['token_type_ids']}")

    print('-'*100)


image_content = {"type": "image", "image": "https://www.lvisdataset.org/assets/images/examples/birdfeeder.jpg"}
image_content_2 = {"type": "image", "image": "http://images.cocodataset.org/val2017/000000039769.jpg"}
image_content_3 = {"type": "image", "image": "https://www.lvisdataset.org/assets/images/examples/teacup.jpg"}
text_content = {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"}

# this is an example multimodal conversation
conversation = [
    {
      "role": "user",
      "content": [
          image_content,
          text_content,
        ],
    },
]

no_image_conversation = [
    {
      "role": "user",
      "content": [
          text_content,
        ],
    },
]

# single turn. multiimage conversation
mi_conversation = [
    {
        "role": "user",
        # two different images and one text
        "content": [
            image_content,
            image_content_2,
            text_content,
        ],
    },
]

# multiturn. multiimage conversation
multiturn_conversation = [
    {
        "role": "user",
        "content": [
            image_content,
            image_content_2,
            text_content,
        ],
    },
    {
        "role": "assistant",
        "content": "I dont know"
    },
    {
        "role": "user",
        "content": [
            image_content_3,
            {"type": "text", "text": "What is this image about?"}
        ],
    },
]

def get_all_images(conversation):
    '''
    Get all the images from the conversation.
    '''
    return [downscale_image(resolve_to_image(content["image"])) for msg in conversation for content in msg["content"] if isinstance(content, dict) and content["type"] == "image"]

# downsample all images
images_conversation = get_all_images(conversation)
images_mi_conversation = get_all_images(mi_conversation)
images_multiturn_conversation = get_all_images(multiturn_conversation)

vlms = [
        # "llava-hf/llava-1.5-7b-hf", 
        # "llava-hf/llava-v1.6-mistral-7b-hf",
        # "llava-hf/llava-v1.6-34b-hf", 
        # "Qwen/Qwen2.5-VL-3B-Instruct", 
        # "google/gemma-3-4b-it",
        # "Qwen/Qwen2-VL-2B",
        # "Qwen/Qwen2.5-Omni-7B",
        # "OpenGVLab/InternVL-Chat-V1-2",
        # "OpenGVLab/InternVL2_5-2B",       # processor does not work 
        # "OpenGVLab/InternVL3-1B-hf",
        # "allenai/Molmo-7B-D-0924",  # no chat template
        #"meta-llama/Llama-4-Scout-17B-16E-Instruct",  # model runs oom
        # "microsoft/Phi-4-multimodal-instruct",
        # "deepseek-ai/deepseek-vl2-tiny",   # no chat template
        # "llava-hf/llava-onevision-qwen2-7b-ov-hf"  # vision
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        # "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        # "llava-hf/llava-interleave-qwen-0.5b-hf",
        # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        # "openai/whisper-large-v3"
]

def get_processor_keys(processor):
    '''
    Print the keys of the processor.
    '''
    all_attrs = set()
    if hasattr(processor, "image_processor"):
        print(f"\tImage Processor: {processor.image_processor.model_input_names}")
        all_attrs.update(processor.image_processor.model_input_names)
    if hasattr(processor, "tokenizer"):
        print(f"\tTokenizer: {processor.tokenizer.model_input_names}")
        all_attrs.update(processor.tokenizer.model_input_names)
    if hasattr(processor, "video_processor"):
        print(f"\tVideo Processor: {processor.video_processor.model_input_names}")
        all_attrs.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        print(f"\tFeature Extractor: {processor.feature_extractor.model_input_names}")
        all_attrs.update(processor.feature_extractor.model_input_names)
    all_attrs.update(processor.model_input_names)
    all_attrs.difference_update(set(processor.tokenizer.model_input_names))
    print(f"All estimated (multimodal) attributes: {all_attrs}")

    return all_attrs

for vlm in vlms:
    print("="*100)
    print(f"Testing {vlm}")
    # processor = AutoProcessor.from_pretrained(vlm)
    processor = processor_dict[vlm]
    print(f"Using {processor.__name__} for {vlm}")
    processor = processor.from_pretrained(vlm, trust_remote_code=True)

    print(f"Processor: {processor}")

    #summarize_conversation_process(processor, no_image_conversation, [], "Single Turn No Image")
    summarize_conversation_process(processor, conversation, images_conversation, "Single Turn Single Image")
    summarize_conversation_process(processor, mi_conversation, images_mi_conversation, "Single Turn Multi Image")
    try:
        summarize_conversation_process(processor, multiturn_conversation, images_multiturn_conversation, "Multi Turn Multi Image")
    except Exception as e:
        print(f"Error: {e}")

    print("-"*100)
    print("Processor Keys:")
    get_processor_keys(processor)

    print("="*100)
    print("\n")
