import math
import torch

import comfy.model_management as mm
import comfy.model_base
import comfy.latent_formats

vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModel
from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

from latent_preview import prepare_callback

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True


def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape
    device = x.device
    dtype = x.dtype

    if F < length:
        y = torch.zeros((B, length, C), dtype=dtype, device=device)
        mask = torch.zeros((B, length), dtype=torch.bool, device=device)
        y[:, :F, :] = x
        mask[:, :F] = True
        return y, mask

    return x[:, :length, :], torch.ones((B, length), dtype=torch.bool, device=device)


class FramePackF1T2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "user_settings": ("UserSettings",),
                "positive": ("Timestamped_Conditioning",),
                "negative": ("Conditioning",),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
            },
            "optional": {
                "sampler_settings": ("SamplerSettings",),
            }
        }
    
    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackF1T2V"

    def process(self, model, positive, negative, user_settings, use_teacache, teacache_rel_l1_thresh, sampler_settings = None):

        # --- Extract data from positive_timed_data --- 
        positive_timed_list = positive["sections"]
        total_second_length = positive["total_duration"]
        latent_window_size = positive["window_size"]
        prompt_blend_sections = positive["blend_sections"]
        print(f"Received - Total Duration: {total_second_length}s, Window Size: {latent_window_size}, Blend Sections: {prompt_blend_sections}")

        # --- F1 Model Type Assumption ---
        # We assume the model loaded into this node is the F1 type.

        # Calculate total sections based on time and window size
        section_frame_duration = latent_window_size * 4 - 3
        if section_frame_duration <= 0: section_frame_duration = 1
        fps = 30 # Assume 30 fps
        section_duration_sec = section_frame_duration / float(fps)
        if section_duration_sec <= 0: section_duration_sec = 1.0 / fps

        # Calculate total sections needed to cover the duration
        total_latent_sections = int(math.ceil(total_second_length / section_duration_sec))
        total_latent_sections = max(total_latent_sections, 1)
        print(f"Total latent sections calculated: {total_latent_sections} (Duration: {total_second_length}s, Section time: {section_duration_sec:.3f}s)")

        # Comfy clean up

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Settings recovery

        t2v_width = user_settings["width"]
        t2v_height = user_settings["height"]
        steps = user_settings["steps"]
        seed = user_settings["seed"]
        
        if sampler_settings is not None:
            cfg = sampler_settings["cfg"]
            guidance_scale = sampler_settings["guidance_scale"]
            shift = sampler_settings["shift"]
            gpu_memory_preservation = sampler_settings["gpu_memory_preservation"]
            sampler = sampler_settings["sampler"]
            denoise_strength = sampler_settings["denoise_strength"]
        else:
            cfg = 1.0
            guidance_scale = 10.0
            shift = 0.0
            gpu_memory_preservation = 6.0
            sampler = 'unipc_bh1'
            denoise_strength = 1.0

        # T2V empty latent
        latent_channels = getattr(transformer.config, 'in_channels', 16)
        latent_tensor = torch.zeros([1, latent_channels, 1, t2v_height // 8, t2v_width // 8])
        #latent_tensor = latent_tensor * vae_scaling_factor

        B, C, T, H, W = latent_tensor.shape
        print(f"Latent dimensions: B={B}, C={C}, T={T}, H={H}, W={W}")

        # --- Conditioning Setup ---
        # Negative conditioning
        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(dtype=base_dtype, device=device)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(dtype=base_dtype, device=device)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        else:
            # Need dummy tensors with correct shape and device
            if positive_timed_list:
                try:
                    # Structure: List[Tuple[float, float, List[List[Union[torch.Tensor, Dict]]]]]
                    first_pos_cond = positive_timed_list[0][2][0][0]
                    first_pos_pooled = positive_timed_list[0][2][0][1]["pooled_output"]
                    dummy_llama_vec = first_pos_cond.to(device)
                    dummy_clip_l_pooler = first_pos_pooled.to(device)
                    llama_vec_n = torch.zeros_like(dummy_llama_vec)
                    clip_l_pooler_n = torch.zeros_like(dummy_clip_l_pooler)
                    llama_vec_n_padded, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                    llama_vec_n = llama_vec_n_padded
                except Exception as e:
                    print(f"Error accessing positive_timed_list for negative shape: {e}. Creating fallback zero tensors.")
                    # Fallback zero tensors if list structure is unexpected
                    llama_vec_n = torch.zeros((B, 512, 4096), dtype=base_dtype, device=device) # Guessing shape based on llama
                    llama_attention_mask_n = torch.ones((B, 512), dtype=torch.long, device=device)
                    clip_l_pooler_n = torch.zeros((B, 1280), dtype=base_dtype, device=device) # Guessing shape based on clip-l
            else:
                 print("Warning: positive_timed_list is empty. Cannot determine negative shape. Creating fallback zero tensors.")
                 llama_vec_n = torch.zeros((B, 512, 4096), dtype=base_dtype, device=device)
                 llama_attention_mask_n = torch.ones((B, 512), dtype=torch.long, device=device)
                 clip_l_pooler_n = torch.zeros((B, 1280), dtype=base_dtype, device=device)

        # Positive conditioning: Handled inside the loop based on time.
        # --- End Conditioning Setup ---

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3 # Frames generated per step

        # F1 History Latents Initialization
        history_latents = torch.zeros(size=(B, 16, 16 + 2 + 1, H, W), dtype=torch.float32).cpu()

        # F1: Start with the initial latent frame
        history_latents = torch.cat([latent_tensor.to(history_latents)], dim=2)
        total_generated_latent_frames = 1 # F1: Start count at 1, representing the initial frame

        # F1 Latent Paddings (determines number of generation steps)
        latent_paddings = [1] * (total_latent_sections - 1) + [0]
        latent_paddings_list = latent_paddings.copy() # For vid2vid indexing
        
        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        #from latent_preview import prepare_callback # Moved to top
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        for i, latent_padding in enumerate(latent_paddings):
            print(f"Sampling Section {i+1}/{total_latent_sections}, latent_padding: {latent_padding}")
            is_last_section = latent_padding == 0

            # F1 logic doesn't seem to use embed interpolation within the loop
            image_encoder_last_hidden_state = None

            # --- Determine Current Positive Conditioning --- 
            # Calculate current time position based on the *start* of the section being generated
            current_time_position = i * section_duration_sec
            current_time_position = max(0.0, current_time_position)
            print(f"  Current time position: {current_time_position:.3f}s")

            active_section_index = -1
            if not positive_timed_list:
                 print("Error: positive_timed_list is empty! Cannot sample.")
                 # Handle error appropriately - maybe return black frames or raise exception?
                 # Returning empty/zeros for now
                 return {"samples": torch.zeros_like(latent_tensor) / vae_scaling_factor},

            for idx, (start_sec, end_sec, _) in enumerate(positive_timed_list):
                # Check if current_time_position falls within [start_sec, end_sec)
                if start_sec <= current_time_position + 1e-4 and current_time_position < end_sec - 1e-4:
                    active_section_index = idx
                    # print(f"  Found active prompt section index: {active_section_index} ({start_sec:.2f}s - {end_sec:.2f}s)")
                    break
            else:
                # If no section matches exactly, check edge cases
                if math.isclose(current_time_position, positive_timed_list[-1][1], abs_tol=1e-4):
                     active_section_index = len(positive_timed_list) - 1
                     # print(f"  Time matches end of last section. Using index: {active_section_index}")
                elif current_time_position >= positive_timed_list[-1][1] - 1e-4:
                     active_section_index = len(positive_timed_list) - 1
                     # print(f"  Time past end of last section. Using index: {active_section_index}")
                elif current_time_position < positive_timed_list[0][0] + 1e-4:
                     active_section_index = 0
                     # print(f"  Time before first section. Using index: 0")
                else: # Final fallback if list exists but no match (should be rare)
                    active_section_index = len(positive_timed_list) - 1
                    print(f"  Warning: No exact time match found, using last section index: {active_section_index}")

            print(f"  Selected active prompt index: {active_section_index}")

            # --- Blending Logic --- 
            blend_alpha = 0.0
            prev_section_idx_for_blend = active_section_index
            next_section_idx_for_blend = active_section_index
            current_active_conditioning_tensor = positive_timed_list[active_section_index][2][0][0]

            # Find the index in the original list corresponding to the *start* of the next *different* conditioning
            next_prompt_change_section_start_index = -1
            next_prompt_change_start_time = -1.0
            for k in range(active_section_index + 1, len(positive_timed_list)):
                # Compare the actual conditioning data (tensors)
                if not torch.equal(positive_timed_list[k][2][0][0], current_active_conditioning_tensor):
                    next_prompt_change_start_time = positive_timed_list[k][0]
                    next_prompt_change_section_start_index = int(round(next_prompt_change_start_time / section_duration_sec))
                    prev_section_idx_for_blend = active_section_index # The prompt active before the change
                    next_section_idx_for_blend = k # The prompt active after the change
                    # print(f"  Next prompt change detected at section index ~{next_prompt_change_section_start_index} (time {next_prompt_change_start_time:.2f}s)")
                    break

            # Check if we are within the blend window leading up to the change
            if prompt_blend_sections > 0 and next_prompt_change_section_start_index != -1:
                blend_start_section_idx = next_prompt_change_section_start_index - prompt_blend_sections
                current_physical_section_idx = i # Use the actual loop iteration index

                if current_physical_section_idx >= blend_start_section_idx and current_physical_section_idx < next_prompt_change_section_start_index:
                    blend_progress = (current_physical_section_idx - blend_start_section_idx + 1) / float(prompt_blend_sections)
                    blend_alpha = max(0.0, min(1.0, blend_progress))
                    print(f"  Blending prompts: Section Index {current_physical_section_idx}, Blend Alpha: {blend_alpha:.3f}")
                # No explicit 'else if >= next_prompt_change...' needed, blend_alpha remains 0 if not in window

            # --- End Blending Logic ---

            # Get the conditioning tensors
            if blend_alpha > 0 and prev_section_idx_for_blend != next_section_idx_for_blend:
                # Ensure indices are valid before accessing
                if 0 <= prev_section_idx_for_blend < len(positive_timed_list) and 0 <= next_section_idx_for_blend < len(positive_timed_list):
                    cond_prev = positive_timed_list[prev_section_idx_for_blend][2][0][0].to(dtype=base_dtype, device=device)
                    pooled_prev = positive_timed_list[prev_section_idx_for_blend][2][0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                    cond_next = positive_timed_list[next_section_idx_for_blend][2][0][0].to(dtype=base_dtype, device=device)
                    pooled_next = positive_timed_list[next_section_idx_for_blend][2][0][1]['pooled_output'].to(dtype=base_dtype, device=device)

                    # Pad tensors before lerp
                    padded_cond_prev, mask_prev = crop_or_pad_yield_mask(cond_prev, length=512)
                    padded_cond_next, mask_next = crop_or_pad_yield_mask(cond_next, length=512)

                    llama_vec = torch.lerp(padded_cond_prev, padded_cond_next, blend_alpha)
                    clip_l_pooler = torch.lerp(pooled_prev, pooled_next, blend_alpha) # Poolers assumed same shape
                    llama_attention_mask = mask_prev # Use mask from the first part of lerp
                else:
                     print(f"Warning: Invalid blend indices ({prev_section_idx_for_blend}, {next_section_idx_for_blend}). Using non-blended active prompt.")
                     # Fallback to non-blended active prompt
                     selected_positive = positive_timed_list[active_section_index][2]
                     llama_vec = selected_positive[0][0].to(dtype=base_dtype, device=device)
                     clip_l_pooler = selected_positive[0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                     llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            else:
                # Use the selected active conditioning directly
                selected_positive = positive_timed_list[active_section_index][2]
                llama_vec = selected_positive[0][0].to(dtype=base_dtype, device=device)
                clip_l_pooler = selected_positive[0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

            # --- End Determine Current Positive Conditioning ---

            # F1 Indices Calculation
            effective_window_size = int(latent_window_size)
            indices = torch.arange(0, sum([1, 16, 2, 1, effective_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, effective_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # F1 Clean Latents Calculation
            required_history_len = 16 + 2 + 1 # Need 19 previous frames
            available_history_len = history_latents.shape[2]

            if available_history_len < required_history_len:
                 print(f"Warning: Not enough history frames ({available_history_len}) for clean latents (needed {required_history_len}). Padding with zeros.")
                 # Pad history_latents at the beginning with zeros to meet required length
                 padding_needed = required_history_len - available_history_len
                 padding_shape = list(history_latents.shape)
                 padding_shape[2] = padding_needed
                 zero_padding = torch.zeros(padding_shape, dtype=history_latents.dtype, device=history_latents.device)
                 padded_history = torch.cat([zero_padding, history_latents], dim=2)
                 clean_latents_4x, clean_latents_2x, clean_latents_1x = padded_history[:, :, -required_history_len:, :, :].split([16, 2, 1], dim=2)
            else:
                 # Take the last 19 frames from history
                 clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -required_history_len:, :, :].split([16, 2, 1], dim=2)

            # Always prepend the original start_latent (frame 0) to clean_latents_1x (the most recent history frame)
            clean_latents = torch.cat([latent_tensor.to(history_latents.device, dtype=history_latents.dtype), clean_latents_1x], dim=2)

            # vid2vid WIP (Using F1's method based on section index 'i')
            input_init_latents = None
            #if initial_samples is not None:
            #    total_length = initial_samples.shape[2]
            #    # Use loop index 'i' for progress, mapping it to the vid2vid timeline
            #    progress = i / (total_latent_sections - 1) if total_latent_sections > 1 else 0
            #    start_idx = int(progress * max(0, total_length - effective_window_size))
            #    end_idx = min(start_idx + effective_window_size, total_length)
            #    # print(f"vid2vid (F1 logic) - Iteration {i}, Progress {progress:.2f}, Slice [{start_idx}:{end_idx}] of {total_length}")
            #    if start_idx < end_idx:
            #        input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device)
            #    else:
            #         print("vid2vid - Warning: Calculated slice is empty.")

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents,
                    strength=denoise_strength,
                    width=W * 8,
                    height=H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            # F1 History Latents Update: Append new frames generated in this step
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            # Increment total frame count by the number of newly generated frames
            total_generated_latent_frames += generated_latents.shape[2]

            # F1 Real History Latents Selection: Take from the end, ensuring we have `total_generated_latent_frames` count
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        # Ensure final output has the expected length (or close to it)
        final_frame_count = real_history_latents.shape[2]
        expected_latent_frames = total_generated_latent_frames # F1 should generate frame by frame
        print(f"Final latent frames: {final_frame_count} (Expected based on generation: {expected_latent_frames})")

        return {"samples": real_history_latents / vae_scaling_factor},            