class FramePackF1T2VUserSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 8, "step": 8, "tooltip": "Output Width."}),
                "height": ("INT", {"default": 640, "min": 8, "step": 8, "tooltip": "Output Height."}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of processing steps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("UserSettings",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "user_settings_packer"
    CATEGORY = "FramePackF1T2V"

    def user_settings_packer(self, width, height, steps, seed):
        settings = {
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed
        }
        return (settings,)