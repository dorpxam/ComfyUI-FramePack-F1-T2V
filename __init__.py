from .nodes.framepack_user_settings_node import FramePackF1T2VUserSettings
from .nodes.framepack_sampler_settings_node import FramePackF1T2VSamplerSettings
from .nodes.framepack_sampler_node import FramePackF1T2VSampler
from .nodes.framepack_timestamps_prompt_node import FramePackF1T2VTextEncode
from .nodes.framepack_lora_stack_node import FramePackF1T2VLoraStack

NODE_CLASS_MAPPINGS = {
    "FramePackF1T2VSampler": FramePackF1T2VSampler,
    "FramePackF1T2VUserSettings": FramePackF1T2VUserSettings,
    "FramePackF1T2VSamplerSettings": FramePackF1T2VSamplerSettings,
    "FramePackF1T2VTextEncode": FramePackF1T2VTextEncode,
    "FramePackF1T2VLoraStack": FramePackF1T2VLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackF1T2VSampler": "FramePack-F1-T2V Sampler",
    "FramePackF1T2VUserSettings": "FramePack-F1-T2V User Settings",
    "FramePackF1T2VSamplerSettings": "FramePack-F1-T2V Sampler Settings",
    "FramePackF1T2VTextEncode": "FramePack-F1-T2V Prompt Encoder",
    "FramePackF1T2VLoraStack": "FramePack-F1-T2V Lora Stack",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]