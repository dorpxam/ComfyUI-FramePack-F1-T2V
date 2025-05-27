import re
import math
import torch

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union


@dataclass
class PromptSection:
    prompt: str
    start_time: float = 0.0  # in seconds
    end_time: Optional[float] = None  # in seconds, None means until the end


def snap_to_section_boundaries(prompt_sections: List[PromptSection], latent_window_size: int, fps: int = 30) -> List[PromptSection]:
    
    section_frame_duration = latent_window_size * 4 - 3
    if section_frame_duration <= 0: 
        section_frame_duration = 1

    section_duration_sec = section_frame_duration / float(fps)
    if section_duration_sec <= 1e-5: 
        section_duration_sec = 1.0 / fps # Avoid zero or near-zero duration

    aligned_sections = []

    for section in prompt_sections:
        aligned_start = round(section.start_time / section_duration_sec) * section_duration_sec
        aligned_end = None

        if section.end_time is not None:
            aligned_end = round(section.end_time / section_duration_sec) * section_duration_sec
            if aligned_end <= aligned_start + 1e-5: # Ensure minimum duration
                aligned_end = aligned_start + section_duration_sec

        aligned_sections.append(PromptSection(
            prompt=section.prompt,
            start_time=aligned_start,
            end_time=aligned_end
        ))

    return aligned_sections


def parse_timestamped_prompt(prompt_text: str, total_duration: float, latent_window_size: int = 9) -> List[PromptSection]:
    sections = []
    # Corrected Regex: Catches [Xs: text] or [Xs-Ys: text]
    timestamp_pattern = r'\[\s*(\d+(?:\.\d+)?s)\s*(?:-\s*(\d+(?:\.\d+)?s)\s*)?:\s*(.*?)\s*\]'
    matches = list(re.finditer(timestamp_pattern, prompt_text))
    last_end_index = 0

    if not matches:
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]
    
    for match in matches:
        plain_text_before = prompt_text[last_end_index:match.start()].strip()
        current_start_time_str = match.group(1)
        current_start_time = float(current_start_time_str.rstrip('s'))
        if plain_text_before:
            previous_end_time = sections[-1].end_time if sections and sections[-1].end_time is not None else (sections[-1].start_time if sections else 0.0)
            if current_start_time > previous_end_time + 1e-5:
                sections.append(PromptSection(prompt=plain_text_before, start_time=previous_end_time, end_time=current_start_time))
            elif not sections and current_start_time > 1e-5:
                sections.append(PromptSection(prompt=plain_text_before, start_time=0.0, end_time=current_start_time))
        end_time_str = match.group(2)
        section_text = match.group(3).strip()
        start_time = current_start_time
        end_time = float(end_time_str.rstrip('s')) if end_time_str else None
        sections.append(PromptSection(prompt=section_text, start_time=start_time, end_time=end_time))
        last_end_index = match.end()

    plain_text_after = prompt_text[last_end_index:].strip()
    
    if plain_text_after:
        previous_end_time = sections[-1].end_time if sections and sections[-1].end_time is not None else sections[-1].start_time
        if total_duration > previous_end_time + 1e-5:
            sections.append(PromptSection(prompt=plain_text_after, start_time=previous_end_time, end_time=None))
    
    if not sections:
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]
    
    sections.sort(key=lambda x: x.start_time)

    # Sanitize and Fill Gaps/Set End Times
    sanitized_sections = []
    current_time = 0.0
    for i, section in enumerate(sections):
        section_start = max(current_time, section.start_time) # Ensure monotonic increase
        section_start = min(section_start, total_duration) # Clamp to total duration

        # Fill gap if needed
        if section_start > current_time + 1e-5:
            filler_prompt = sanitized_sections[-1].prompt if sanitized_sections else "" # Use previous prompt
            sanitized_sections.append(PromptSection(prompt=filler_prompt, start_time=current_time, end_time=section_start))

        # Determine end time
        section_end = section.end_time
        if section_end is None:
            if i + 1 < len(sections):
                next_start = max(section_start, sections[i+1].start_time) # Ensure next start is after current start
                section_end = min(next_start, total_duration) # End before next or at total duration
            else:
                section_end = total_duration # Last section ends at total duration
        else:
            section_end = min(max(section_start, section_end), total_duration) # Clamp user-defined end

        # Add the section if it has duration
        if section_end > section_start + 1e-5:
            sanitized_sections.append(PromptSection(prompt=section.prompt, start_time=section_start, end_time=section_end))
            current_time = section_end # Update current time marker
        elif i == len(sections) - 1 and math.isclose(section_start, total_duration): # Allow point at the end? No, remove.
            pass

    if not sanitized_sections:
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]
    
    # Snap timestamps to boundaries
    aligned_sections = snap_to_section_boundaries(sanitized_sections, latent_window_size)

    # Merge identical consecutive prompts after snapping
    merged_sections = []

    if not aligned_sections: 
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]
    
    current_merged = aligned_sections[0]

    for i in range(1, len(aligned_sections)):
        next_sec = aligned_sections[i]
        # Merge if prompts are identical and sections are contiguous (or very close after snapping)
        if next_sec.prompt == current_merged.prompt and abs(next_sec.start_time - current_merged.end_time) < 0.01:
            current_merged.end_time = next_sec.end_time # Extend the end time
        else:
            current_merged.end_time = max(current_merged.start_time, current_merged.end_time)
            if current_merged.start_time < current_merged.end_time - 1e-5:
                merged_sections.append(current_merged)
            current_merged = next_sec

    current_merged.end_time = max(current_merged.start_time, current_merged.end_time)
    if current_merged.start_time < current_merged.end_time - 1e-5:
        merged_sections.append(current_merged)

    if not merged_sections:
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]
    
    print("Parsed Prompt Sections:")
    for sec in merged_sections: 
        print(f"  [{sec.start_time:.3f}s - {sec.end_time:.3f}s]: {sec.prompt}")

    return merged_sections


class FramePackF1T2VTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive prompt, use [Xs: prompt] or [Xs-Ys: prompt] for timed sections."}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Negative prompt."}),
                "total_second_length": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 1200.0, "step": 0.1, "tooltip": "Expected total video duration in seconds for timestamp calculation."}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The latent window size used by the sampler for timestamp boundary snapping."}),
                "prompt_blend_sections": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Number of latent sections (windows) over which to blend prompts when they change. 0 disables blending."}),
            }
        }
    
    RETURN_TYPES = ("Timestamped_Conditioning", "Conditioning",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"
    CATEGORY = "FramePackF1T2V"
    DESCRIPTION = "Encodes text prompts with optional timestamps for timed conditioning."

    def encode(self, clip, positive, negative, total_second_length, latent_window_size, prompt_blend_sections):

        prompt_sections = parse_timestamped_prompt(positive, total_second_length, latent_window_size)
        unique_prompts = sorted(list(set(section.prompt for section in prompt_sections)))
        encoded_prompts: Dict[str, List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]] = {}
        first_cond, first_pooled = None, None

        print(f"FramePackF1T2VTextEncode: Encoding {len(unique_prompts)} unique prompts.")
        
        for i, prompt in enumerate(unique_prompts):
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            if i == 0:
                first_cond, first_pooled = cond, pooled
            encoded_prompts[prompt] = [[cond, {"pooled_output": pooled}]]

        positive_timed_list: List[Tuple[float, float, List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]]] = []
        for section in prompt_sections:
            if section.prompt in encoded_prompts:
                encoded_cond = encoded_prompts[section.prompt]
                positive_timed_list.append((section.start_time, section.end_time, encoded_cond))
            else:
                print(f"Warning: Prompt '{section.prompt}' not found in encoded prompts. Skipping section.")

        if negative:
            tokens_neg = clip.tokenize(negative)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative = [[cond_neg, {"pooled_output": pooled_neg}]]
        elif first_cond is not None:
            negative = [[torch.zeros_like(first_cond), {"pooled_output": torch.zeros_like(first_pooled)}]]
        else:
            print("FramePackF1T2VTextEncode: Error - Cannot create empty negative conditioning, no positive prompts found and fallback failed.")
            try:
                tokens = clip.tokenize("")
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                negative = [[torch.zeros_like(cond), {"pooled_output": torch.zeros_like(pooled)}]]
            except Exception as e:
                print(f"Fallback negative shape guess failed: {e}")
                # Minimal fallback guess
                negative = [[torch.zeros((1, 77, 768)), {"pooled_output": torch.zeros((1, 768))}]]

        # Package results into a dictionary
        timed_data = {
            "sections": positive_timed_list,
            "total_duration": total_second_length,
            "window_size": latent_window_size,
            "blend_sections": prompt_blend_sections
        }
        return (timed_data, negative)