import folder_paths

def makeloraentry(lora, strength, fuse_lora):
    return {
        "path": folder_paths.get_full_path("loras", lora),
        "strength": strength,
        "name": lora.split(".")[0],
        "fuse_lora": fuse_lora,
    }

class FramePackF1T2VLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_01": (['None'] + lora_list, ),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_01": ("BOOLEAN", {"default": False}),

                "lora_02": (['None'] + lora_list, ),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_02": ("BOOLEAN", {"default": False}),

                "lora_03": (['None'] + lora_list, ),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_03": ("BOOLEAN", {"default": False}),

                "lora_04": (['None'] + lora_list, ),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_04": ("BOOLEAN", {"default": False}),

                "lora_05": (['None'] + lora_list, ),
                "strength_05": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_05": ("BOOLEAN", {"default": False}),

                "lora_06": (['None'] + lora_list, ),
                "strength_06": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_06": ("BOOLEAN", {"default": False}),

                "lora_07": (['None'] + lora_list, ),
                "strength_07": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_07": ("BOOLEAN", {"default": False}),

                "lora_08": (['None'] + lora_list, ),
                "strength_08": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "activate_08": ("BOOLEAN", {"default": False}),
                
                "fuse_lora": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
            },
            "optional": {
                "prev_lora": ("FPLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("FPLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "FramePackF1T2V"

    def getlorapath(self, lora_01, strength_01, activate_01, lora_02, strength_02, activate_02, lora_03, strength_03, activate_03, lora_04, strength_04, activate_04,
                          lora_05, strength_05, activate_05, lora_06, strength_06, activate_06, lora_07, strength_07, activate_07, lora_08, strength_08, activate_08, prev_lora=None, fuse_lora=False):
        
        loras_list = []

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        if lora_01 != "None" and strength_01 != 0 and activate_01:
            loras_list.append(makeloraentry(lora_01, strength_01, fuse_lora))
        if lora_02 != "None" and strength_02 != 0 and activate_02:
            loras_list.append(makeloraentry(lora_02, strength_02, fuse_lora))
        if lora_03 != "None" and strength_03 != 0 and activate_03:
            loras_list.append(makeloraentry(lora_03, strength_03, fuse_lora))
        if lora_04 != "None" and strength_04 != 0 and activate_04:
            loras_list.append(makeloraentry(lora_04, strength_04, fuse_lora))
        if lora_05 != "None" and strength_05 != 0 and activate_05:
            loras_list.append(makeloraentry(lora_05, strength_05, fuse_lora))
        if lora_06 != "None" and strength_06 != 0 and activate_06:
            loras_list.append(makeloraentry(lora_06, strength_06, fuse_lora))
        if lora_07 != "None" and strength_07 != 0 and activate_07:
            loras_list.append(makeloraentry(lora_07, strength_07, fuse_lora))
        if lora_08 != "None" and strength_08 != 0 and activate_08:
            loras_list.append(makeloraentry(lora_08, strength_08, fuse_lora))

        return (loras_list,)