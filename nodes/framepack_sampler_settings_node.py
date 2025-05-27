class FramePackF1T2VSamplerSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"], { "default": 'unipc_bh1' }),
            },
            "optional": {
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("SamplerSettings",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "sampler_settings_packer"
    CATEGORY = "FramePackF1T2V"

    def sampler_settings_packer(self, cfg, guidance_scale, shift, gpu_memory_preservation, sampler, denoise_strength=1.0):
        settings = {
            "cfg": cfg,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "gpu_memory_preservation": gpu_memory_preservation,
            "sampler": sampler,
            "denoise_strength": denoise_strength
        }
        return (settings,)