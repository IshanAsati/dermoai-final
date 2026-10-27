import timm
import torch
from torch import nn
from timm.data import create_transform, resolve_model_data_config
from torchvision import transforms
from torchvision.transforms import InterpolationMode

REGISTRY = {
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "densenet121": "densenet121.ra_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "resnet50": "resnet50.a1_in1k",
}


class MixStyle(nn.Module):
    def __init__(self, probability=0.0, alpha=0.1, epsilon=1e-6):
        super().__init__()
        self.probability = probability
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, features):
        if not self.training or self.probability <= 0 or features.ndim != 4 or len(features) < 2:
            return features
        if torch.rand((), device=features.device) > self.probability:
            return features
        mean = features.mean(dim=(2, 3), keepdim=True)
        standard_deviation = (features.var(dim=(2, 3), keepdim=True, unbiased=False) + self.epsilon).sqrt()
        normalized = (features - mean) / standard_deviation
        permutation = torch.randperm(len(features), device=features.device)
        concentration = torch.tensor(self.alpha, device=features.device, dtype=torch.float32)
        mixing = torch.distributions.Beta(concentration, concentration).sample((len(features),)).to(features.dtype).view(-1, 1, 1, 1)
        mixed_mean = mixing * mean + (1 - mixing) * mean[permutation]
        mixed_standard_deviation = mixing * standard_deviation + (1 - mixing) * standard_deviation[permutation]
        return normalized * mixed_standard_deviation + mixed_mean


def auxiliary_modules(model):
    return [getattr(model, name) for name in ("aux_cancer_head", "aux_family_head") if hasattr(model, name)]


def device_for(config):
    if config.get("device") == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install the CUDA PyTorch build; CPU fallback is disabled")
        return torch.device("cuda")
    if config.get("synthetic") and config.get("device") == "cpu":
        return torch.device("cpu")
    raise ValueError("Explicit synthetic test required for CPU")


def build_model(config, load_pretrained=None):
    name = REGISTRY[config["architecture"]]
    if name not in timm.list_models(pretrained=True, include_tags=True):
        raise ValueError(f"Pretrained checkpoint not registered in this timm version: {name}")
    pretrained = config.get("pretrained", True) if load_pretrained is None else load_pretrained
    arguments = {"img_size": config["image_size"]} if config["architecture"] == "swin_tiny" else {}
    model = timm.create_model(name, pretrained=pretrained, num_classes=len(config["labels"]), **arguments)
    if config.get("auxiliary_heads"):
        model.aux_cancer_head = nn.Linear(model.num_features, 2)
        model.aux_family_head = nn.Linear(model.num_features, 4)
    if config.get("mixstyle_probability", 0.0) > 0:
        model.mixstyle = MixStyle(config["mixstyle_probability"], config.get("mixstyle_alpha", 0.1))
        model._mixstyle_hook = model.stages[0].register_forward_hook(lambda module, inputs, output: model.mixstyle(output))
    mode = config["mode"]
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    for module in auxiliary_modules(model):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if mode == "partial":
        prefixes = config.get("unfreeze_prefixes", [])
        if not prefixes:
            raise ValueError("Partial fine-tuning requires explicit unfreeze_prefixes")
        for prefix in prefixes:
            matched = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
            if not matched:
                raise ValueError(f"No parameters match prefix: {prefix}")
            for parameter in matched:
                parameter.requires_grad = True
    return model


def transforms_for(model, config, training=False):
    data = resolve_model_data_config(model)
    data["input_size"] = (3, config["image_size"], config["image_size"])
    augmentation = config.get("augmentation")
    if augmentation == "conservative_skin":
        if training:
            data.update(scale=(0.85, 1.0), ratio=(0.9, 1.1), hflip=0.5, vflip=0.5, color_jitter=0.1, re_prob=0.0, auto_augment=None)
        else:
            data.update(crop_pct=1.0, crop_mode="border")
    elif augmentation == "aggressive_smartphone_skin":
        if training:
            data.update(
                scale=(0.72, 1.0),
                ratio=(0.82, 1.18),
                hflip=0.5,
                vflip=0.5,
                color_jitter=(0.15, 0.15, 0.1, 0.02),
                color_jitter_prob=0.8,
                gaussian_blur_prob=0.15,
                re_prob=0.0,
                auto_augment=None,
            )
        else:
            data.update(crop_pct=1.0, crop_mode="border")
    transform = create_transform(**data, is_training=training)
    if training and augmentation == "aggressive_smartphone_skin":
        transform.transforms[1:1] = [
            transforms.RandomRotation(20, interpolation=InterpolationMode.BILINEAR, fill=(124, 116, 104)),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.15, interpolation=InterpolationMode.BILINEAR, fill=(124, 116, 104)),
            transforms.RandomApply([transforms.RandomAdjustSharpness(1.5)], p=0.15),
        ]
    return transform


def training_mode(model, mode):
    if mode == "frozen":
        model.eval()
        model.get_classifier().train()
        for module in auxiliary_modules(model):
            module.train()
    else:
        model.train()
        for module in model.modules():
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()
