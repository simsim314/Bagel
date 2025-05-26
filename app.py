import gradio as gr
import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
# from modeling.bagel.qwen2_navit import NaiveCache # NaiveCache might not be directly used in app.py top level
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--model_path", type=str, default="downloads") # Default path

args = parser.parse_args()

# --- START OF MODEL INITIALIZATION (Ensure this block is complete and before device mapping) ---
model_path = args.model_path
print(f"Using model files from: {os.path.abspath(model_path)}")
if not os.path.exists(os.path.join(model_path, "llm_config.json")):
    print(f"ERROR: llm_config.json not found in {model_path}. Please check model download.")
    exit()

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1 # As per original notebook

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

# Initialize empty model shell
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model_shell = SiglipVisionModel(vit_config) # Renamed to avoid conflict with loaded vit_model if any
    model = Bagel(language_model, vit_model_shell, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

print("Initial model components and configurations prepared.")
# --- END OF MODEL INITIALIZATION ---

# --- START OF MODEL LOADING AND MULTI-GPU DEVICE MAPPING (Corrected) ---
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")

num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("WARNING: No GPUs detected by PyTorch. Model will be loaded on CPU.")
    device_map = "cpu"
    # num_llm_layers already available from llm_config loaded above
else:
    print(f"Preparing manual device map for {num_gpus} GPU(s).")
    num_llm_layers = llm_config.num_hidden_layers # llm_config IS DEFINED
    print(f"Number of LLM layers: {num_llm_layers}")

    device_map = {}

    # --- LLM Core Distribution ---
    llm_embed_gpu = 0 # This will be our "interaction_gpu" for combining modalities
    device_map["language_model.model.embed_tokens"] = llm_embed_gpu

    layers_per_gpu_base = num_llm_layers // num_gpus
    remainder_layers = num_llm_layers % num_gpus
    current_layer_idx = 0
    for i in range(num_gpus):
        num_layers_on_this_gpu = layers_per_gpu_base + (1 if i < remainder_layers else 0)
        for _ in range(num_layers_on_this_gpu):
            if current_layer_idx < num_llm_layers:
                device_map[f"language_model.model.layers.{current_layer_idx}"] = i
                current_layer_idx += 1
            else:
                break
        if current_layer_idx >= num_llm_layers:
            break

    final_llm_components_gpu = (num_gpus - 1) if num_gpus > 0 else 0
    if num_llm_layers > 0:
        last_layer_device_key = f"language_model.model.layers.{num_llm_layers - 1}"
        if last_layer_device_key in device_map:
            final_llm_components_gpu = device_map[last_layer_device_key]
        elif num_gpus > 0 :
             final_llm_components_gpu = (num_gpus - 1)

    device_map["language_model.model.norm"] = final_llm_components_gpu
    device_map["language_model.lm_head"] = final_llm_components_gpu
    if hasattr(model.language_model.model, 'norm_moe_gen'):
        device_map["language_model.model.norm_moe_gen"] = final_llm_components_gpu

    # --- Vision Components ---
    # vit_model itself can be large, place it on a GPU (e.g., the last one if multi-GPU)
    vit_processing_gpu = (num_gpus - 1) if num_gpus > 1 else 0 # e.g. cuda:1 for 2 GPUs
    device_map["vit_model"] = vit_processing_gpu

    # **CRITICAL FIX in this section:**
    # Place vit_pos_embed and connector on the GPU where their
    # output will be combined with other embeddings for the LLM (i.e., llm_embed_gpu).
    interaction_gpu = llm_embed_gpu # This is cuda:0

    device_map["vit_pos_embed"] = interaction_gpu # To GPU 0
    device_map["connector"] = interaction_gpu     # To GPU 0
    print(f"ViT on GPU {vit_processing_gpu}. Connector and ViT Positional Embeddings on GPU {interaction_gpu}.")


    # --- Other Bagel-specific components ---
    device_map["time_embedder"] = llm_embed_gpu
    device_map["latent_pos_embed"] = llm_embed_gpu
    device_map["vae2llm"] = llm_embed_gpu
    device_map["llm2vae"] = final_llm_components_gpu

print("Constructed manual device_map:", device_map)

# ... (rest of the loading code: offload_folder, max_memory_dict, load_checkpoint_and_dispatch) ...
# Make sure model_path is defined before load_checkpoint_and_dispatch
# model_path = args.model_path # Should be already set from args

offload_folder = "offload_bagel"
if device_map != "cpu" and any(v == 'disk' for v in device_map.values()):
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder, exist_ok=True)
    print(f"Disk offloading is active. Using offload folder: {offload_folder}")
else:
    print("No disk offloading specified in manual device_map or device_map is 'cpu'.")

if device_map != "cpu" and num_gpus > 0:
    max_memory_dict = {i: "44GiB" for i in range(num_gpus)}
else:
    max_memory_dict = None
print(f"Using max_memory for dispatch: {max_memory_dict}")

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    max_memory=max_memory_dict,
    offload_buffers=True if device_map != "cpu" else False,
    offload_folder=offload_folder if device_map != "cpu" else None,
    dtype=torch.bfloat16,
    force_hooks=True,
).eval()
print("Model loaded and dispatched.")
# --- END OF MODEL LOADING AND MULTI-GPU DEVICE MAPPING ---

# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

# Text to Image function with thinking option and hyperparameters
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, 
                 timestep_shift=3.0, num_timesteps=50, 
                 cfg_renorm_min=1.0, cfg_renorm_type="global", 
                 max_think_token_n=1024, do_sample=False, text_temperature=0.3,
                 seed=0, image_ratio="1:1"):
    # Set seed for reproducibility
    set_seed(seed)

    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768) 
    elif image_ratio == "16:9":
        image_shapes = (576, 1024)
    elif image_ratio == "9:16":
        image_shapes = (1024, 576) 
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    
    # Call inferencer with or without think parameter based on user choice
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", None)


# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False, 
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )
    
    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking, 
                        understanding_output=True, **inference_hyper)
    return result["text"]


# Image Editing function with thinking option and hyperparameters
def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, 
              cfg_img_scale=2.0, cfg_interval=0.0, 
              timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=1.0, 
              cfg_renorm_type="text_channel", max_think_token_n=1024, 
              do_sample=False, text_temperature=0.3, seed=0):
    # Set seed for reproducibility
    set_seed(seed)
    
    if image is None:
        return "Please upload an image.", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Include thinking parameter based on user choice
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", "")


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


# Gradio UI 
with gr.Blocks() as demo:
    gr.Markdown("""
<div>
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="380"/>
</div>
""")

    with gr.Tab("📝 Text to Image"):
        txt_input = gr.Textbox(
            label="Prompt", 
            value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
        )
        
        with gr.Row():
            show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, 
                                   label="Seed", info="0 for random seed, positive for reproducible results")
                    image_ratio = gr.Dropdown(choices=["1:1", "4:3", "3:4", "16:9", "9:16"], 
                                                value="1:1", label="Image Ratio", 
                                                info="The longer size is fixed to 1024")
                    
                with gr.Row():
                    cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                             label="CFG Text Scale", info="Controls how strongly the model follows the text prompt (4.0-8.0)")
                    cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, 
                                           label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                
                with gr.Row():
                    cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                value="global", label="CFG Renorm Type", 
                                                info="If the genrated image is blurry, use 'global'")
                    cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                             label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                            label="Timesteps", info="Total denoising steps")
                    timestep_shift = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, interactive=True,
                                             label="Timestep Shift", info="Higher values for layout, lower for details")
                
                # Thinking parameters in a single row
                thinking_params = gr.Group(visible=False)
                with thinking_params:
                    with gr.Row():
                        do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                    label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                  label="Temperature", info="Controls randomness in text generation")
        
        thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        img_output = gr.Image(label="Generated Image")
        gen_btn = gr.Button("Generate", variant="primary")
        
        # Dynamically show/hide thinking process box and parameters
        def update_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        show_thinking.change(
            fn=update_thinking_visibility,
            inputs=[show_thinking],
            outputs=[thinking_output, thinking_params]
        )
        
        # Process function based on thinking option and hyperparameters
        def process_text_to_image(prompt, show_thinking, cfg_text_scale, 
                                 cfg_interval, timestep_shift, 
                                 num_timesteps, cfg_renorm_min, cfg_renorm_type, 
                                 max_think_token_n, do_sample, text_temperature, seed, image_ratio):
            image, thinking = text_to_image(
                prompt, show_thinking, cfg_text_scale, cfg_interval,
                timestep_shift, num_timesteps, 
                cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed, image_ratio
            )
            return image, thinking if thinking else ""
        
        gr.on(
            triggers=[gen_btn.click, txt_input.submit],
            fn=process_text_to_image,
            inputs=[
                txt_input, show_thinking, cfg_text_scale, 
                cfg_interval, timestep_shift, 
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed, image_ratio
            ],
            outputs=[img_output, thinking_output]
        )

    with gr.Tab("🖌️ Image Edit"):
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_input = gr.Image(label="Input Image", value=load_example_image('test_images/women.jpg'))
                edit_prompt = gr.Textbox(
                    label="Prompt",
                    value="She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."
                )
            
            with gr.Column(scale=1):
                edit_image_output = gr.Image(label="Result")
                edit_thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        
        with gr.Row():
            edit_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    edit_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, interactive=True,
                                        label="Seed", info="0 for random seed, positive for reproducible results")
                    edit_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                                  label="CFG Text Scale", info="Controls how strongly the model follows the text prompt")
                
                with gr.Row():
                    edit_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, interactive=True,
                                                 label="CFG Image Scale", info="Controls how much the model preserves input image details")
                    edit_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                    
                with gr.Row():
                    edit_cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                     value="text_channel", label="CFG Renorm Type", 
                                                     info="If the genrated image is blurry, use 'global")
                    edit_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                  label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    edit_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                                 label="Timesteps", info="Total denoising steps")
                    edit_timestep_shift = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, interactive=True,
                                                  label="Timestep Shift", info="Higher values for layout, lower for details")
                
                
                # Thinking parameters in a single row
                edit_thinking_params = gr.Group(visible=False)
                with edit_thinking_params:
                    with gr.Row():
                        edit_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        edit_max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                         label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        edit_text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                        label="Temperature", info="Controls randomness in text generation")
        
        edit_btn = gr.Button("Submit", variant="primary")
        
        # Dynamically show/hide thinking process box for editing
        def update_edit_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        edit_show_thinking.change(
            fn=update_edit_thinking_visibility,
            inputs=[edit_show_thinking],
            outputs=[edit_thinking_output, edit_thinking_params]
        )
        
        # Process editing with thinking option and hyperparameters
        def process_edit_image(image, prompt, show_thinking, cfg_text_scale, 
                              cfg_img_scale, cfg_interval, 
                              timestep_shift, num_timesteps, cfg_renorm_min, 
                              cfg_renorm_type, max_think_token_n, do_sample, 
                              text_temperature, seed):
            edited_image, thinking = edit_image(
                image, prompt, show_thinking, cfg_text_scale, cfg_img_scale, 
                cfg_interval, timestep_shift, 
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed
            )
            
            return edited_image, thinking if thinking else ""
        
        gr.on(
            triggers=[edit_btn.click, edit_prompt.submit],
            fn=process_edit_image,
            inputs=[
                edit_image_input, edit_prompt, edit_show_thinking, 
                edit_cfg_text_scale, edit_cfg_img_scale, edit_cfg_interval,
                edit_timestep_shift, edit_num_timesteps, 
                edit_cfg_renorm_min, edit_cfg_renorm_type,
                edit_max_think_token_n, edit_do_sample, edit_text_temperature, edit_seed
            ],
            outputs=[edit_image_output, edit_thinking_output]
        )

    with gr.Tab("🖼️ Image Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Input Image", value=load_example_image('test_images/meme.jpg'))
                understand_prompt = gr.Textbox(
                    label="Prompt", 
                    value="Can someone explain what's funny about this meme??"
                )
            
            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Result", lines=20)
        
        with gr.Row():
            understand_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Row():
                understand_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                understand_text_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, interactive=True,
                                                     label="Temperature", info="Controls randomness in text generation (0=deterministic, 1=creative)")
                understand_max_new_tokens = gr.Slider(minimum=64, maximum=4096, value=512, step=64, interactive=True,
                                                   label="Max New Tokens", info="Maximum length of generated text, including potential thinking")
        
        img_understand_btn = gr.Button("Submit", variant="primary")
        
        # Process understanding with thinking option and hyperparameters
        def process_understanding(image, prompt, show_thinking, do_sample, 
                                 text_temperature, max_new_tokens):
            result = image_understanding(
                image, prompt, show_thinking, do_sample, 
                text_temperature, max_new_tokens
            )
            return result
        
        gr.on(
            triggers=[img_understand_btn.click, understand_prompt.submit],
            fn=process_understanding,
            inputs=[
                img_input, understand_prompt, understand_show_thinking,
                understand_do_sample, understand_text_temperature, understand_max_new_tokens
            ],
            outputs=txt_output
        )

    gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://discord.gg/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</div>
""")

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        inbrowser=True,
    )