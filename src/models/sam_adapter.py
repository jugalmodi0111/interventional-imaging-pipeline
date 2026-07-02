"""CoroSAM-style: LiteMedSAM/MedficientSAM + LoRA adapters for prompt-based seg/labeling."""
# TODO: load a lightweight SAM (MobileSAM / EdgeSAM / MedficientSAM checkpoint),
#       wrap with peft.LoraConfig, fine-tune mask decoder on ARCADE point/box prompts.
def build_adapter(base_ckpt, r=8):
    raise NotImplementedError
