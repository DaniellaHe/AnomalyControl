import os
from typing import List

from ano_csm.models import load_model_and_preprocess
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn as nn
from .utils import is_torch2_available, get_generator
from sg_adapter.sg_adapter import ImageProjModel, BlipProjModel

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        SGAttnProcessor2_0 as SGAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, SGAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        # embeds = embeds.to(self.proj.weight.device)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ImageTextFusionAdapter:
    def __init__(self, sd_pipe, image_encoder_path, sg_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = sg_ckpt
        self.num_tokens = num_tokens

        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor",
                                                                          model_type="base", is_eval=True,
                                                                          device=self.device)

        self.pipe = sd_pipe.to(self.device)
        self.set_sg_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_sg_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_sg_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SGAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_sg_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "sg_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("sg_adapter."):
                        state_dict["sg_adapter"][key.replace("sg_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        sg_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        sg_layers.load_state_dict(state_dict["sg_adapter"])

    # @torch.inference_mode()
    # def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    #     if pil_image is not None:
    #         if isinstance(pil_image, Image.Image):
    #             pil_image = [pil_image]
    #         clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    #         clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
    #     else:
    #         clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
    #     multimodal_embeds = self.image_proj_model(clip_image_embeds)
    #     uncond_multimodal_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
    #     return multimodal_embeds, uncond_multimodal_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SGAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_multimodal_embeds(self, ref_pair_embeds):
        ref_image = ref_pair_embeds[0]
        ref_text = ref_pair_embeds[1]
        image = self.vis_processors["eval"](ref_image).unsqueeze(0).to(self.device)
        text_input = self.txt_processors["eval"](ref_text)
        sample = {"image": image, "text_input": [text_input]}
        features_multimodal = self.blip_model.extract_features(sample)
        multimodal_embeds = features_multimodal.multimodal_embeds

        unknown_dim = multimodal_embeds.size(1)

        linear_proj = nn.Linear(in_features=unknown_dim, out_features=4).to(multimodal_embeds.device)

        multimodal_embeds = linear_proj(multimodal_embeds.permute(0, 2, 1)).permute(0, 2, 1)

        uncond_multimodal_embeds = torch.zeros_like(multimodal_embeds)
        return multimodal_embeds, uncond_multimodal_embeds

    def generate(
            self,
            # pil_image=None,
            clip_image_embeds=None,
            ref_pair_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=20,
            **kwargs,
    ):
        self.set_scale(scale)

        multimodal_embeds, uncond_multimodal_embeds = self.get_multimodal_embeds(ref_pair_embeds)

        ref_image = ref_pair_embeds[0]
        num_prompts = 1 if isinstance(ref_image, Image.Image) else len(ref_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        bs_embed, seq_len, _ = multimodal_embeds.shape
        multimodal_embeds = multimodal_embeds.repeat(1, num_samples, 1)
        multimodal_embeds = multimodal_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_multimodal_embeds = uncond_multimodal_embeds.repeat(1, num_samples, 1)
        uncond_multimodal_embeds = uncond_multimodal_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, multimodal_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_multimodal_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class anoSGAdapter:
    def __init__(self, sd_pipe, sg_ckpt, device, num_tokens=4, multimodal_name="blip", scale=1.0):
        self.device = device
        # self.image_encoder_path = image_encoder_path
        self.ip_ckpt = sg_ckpt
        self.num_tokens = num_tokens
        self.scale = scale

        self.pipe = sd_pipe.to(self.device)
        self.set_sg_adapter()

        # load image encoder
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
        #     self.device, dtype=torch.float16
        # )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.blip_proj_model, self.image_proj_model = self.init_proj()

        self.load_sg_adapter()
        if multimodal_name == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip_feature_extractor",
                model_type="base", is_eval=True)
        elif multimodal_name == "blip2":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain", is_eval=True)

    def init_proj(self):
        # image_proj_model = ImageProjModel(
        #     cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
        #     clip_embeddings_dim=768,
        #     # clip_embeddings_dim=self.image_encoder.config.projection_dim,  # 1024
        #     clip_extra_context_tokens=self.num_tokens,  # 4
        # ).to(self.device, dtype=torch.float16)
        blip_proj_model = BlipProjModel(
            input_embeddings_dim=768,
            output_attention_dim=1024).to(self.device, dtype=torch.float16)
        image_proj_model = ImageProjModel(
            # clip_embeddings_dim=768,
            clip_embeddings_dim=1024,  # 1024
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,  # 768
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)
        return blip_proj_model, image_proj_model

    def set_sg_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SGAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=self.scale,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_sg_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "sg_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("sg_adapter."):
                        state_dict["sg_adapter"][key.replace("sg_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.blip_proj_model.load_state_dict(state_dict["blip_proj"])
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        sg_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        sg_layers.load_state_dict(state_dict["sg_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, multimodal_feature=None):
        # if pil_image is not None:
        #     if isinstance(pil_image, Image.Image):
        #         pil_image = [pil_image]
        #     # clip_image: [1, 3, 224 224]
        #     clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        #     # clip_image_embeds: [1, 1024]
        #     clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        # else:
        #     clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        # image_prompt_embeds: [1,4 768]
        multimodal_feature = multimodal_feature.to(self.device, dtype=torch.float16)
        blip_prompt_embeds = self.blip_proj_model(multimodal_feature)
        image_prompt_embeds = self.image_proj_model(blip_prompt_embeds)

        uncond_blip_prompt_embeds = self.blip_proj_model(torch.zeros_like(multimodal_feature))
        uncond_image_prompt_embeds = self.image_proj_model(uncond_blip_prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SGAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        subject_text=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        image_file=None,
        ref_mask=None,
        use_ref_mask=False,
        **kwargs,
    ):
        self.set_scale(scale)

        # [1, 3, 224, 224]
        image_input = self.vis_processors["eval"](pil_image.convert("RGB")).unsqueeze(0)
        subject_text_input = self.txt_processors["eval"](subject_text)
        # sample = {"image": image_input, "text_input": [subject_text_input]}
        # # [1, 4, 768]
        # multimodal_feature = self.blip_model.extract_features(sample).multimodal_embeds
        # multimodal_feature = torch.mean(multimodal_feature, dim=1)

        if use_ref_mask:
            # [1, 3, 768]
            sample = {"image": image_input, "text_input": [subject_text_input], "raw_image": pil_image, "image_file": image_file.replace("/", "_").replace(".png", "")}
            multimodal_feature = self.blip_model.extract_features_with_ref_mask(sample, ref_mask, mode="multimodal", show_att=True).multimodal_embeds
        else:
            # [1, 3, 768]
            sample = {"image": image_input, "text_input": [subject_text_input]}
            multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds

        multimodal_feature = torch.mean(multimodal_feature, dim=1)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = multimodal_feature.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, multimodal_feature=multimodal_feature
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)



        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class anoSGAdapter_inpainting:
    def __init__(self, sd_pipe, sg_ckpt, device, num_tokens=4, multimodal_name="blip", scale=1.0):
        self.device = device
        self.ip_ckpt = sg_ckpt
        self.num_tokens = num_tokens
        self.scale = scale

        self.pipe = sd_pipe.to(self.device)

        self.set_sg_adapter()

        # image proj model
        self.blip_proj_model, self.image_proj_model = self.init_proj()

        self.load_sg_adapter()
        if multimodal_name == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip_feature_extractor",
                model_type="base", is_eval=True)
        elif multimodal_name == "blip2":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain", is_eval=True)

    def init_proj(self):
        blip_proj_model = BlipProjModel(
            input_embeddings_dim=768,
            output_attention_dim=1024).to(self.device, dtype=torch.float16)
        image_proj_model = ImageProjModel(
            # clip_embeddings_dim=768,
            clip_embeddings_dim=1024,  # 1024
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,  # 768
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)
        return blip_proj_model, image_proj_model

    def set_sg_adapter(self):
        unet = self.pipe.unet

        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SGAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=self.scale,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)


        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_sg_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "sg_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("sg_adapter."):
                        state_dict["sg_adapter"][key.replace("sg_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.blip_proj_model.load_state_dict(state_dict["blip_proj"])
        self.image_proj_model.load_state_dict(state_dict["image_proj"])


        sg_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        sg_layers.load_state_dict(state_dict["sg_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, multimodal_feature=None):
        multimodal_feature = multimodal_feature.to(self.device, dtype=torch.float16)

        blip_prompt_embeds = self.blip_proj_model(multimodal_feature)
        image_prompt_embeds = self.image_proj_model(blip_prompt_embeds)

        uncond_blip_prompt_embeds = self.blip_proj_model(torch.zeros_like(multimodal_feature))
        uncond_image_prompt_embeds = self.image_proj_model(uncond_blip_prompt_embeds)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SGAttnProcessor):
                attn_processor.scale = scale

    # images = sg_model.generate(image=image, mask_image=mask, prompt=prompt,
    #                                pil_image=ref_image, ref_mask=mask, subject_text=ref_text,
    #                                num_samples=num_samples, num_inference_steps=50,
    #                                seed=42,  strength=0.7,
    #                                image_file=image_id, use_ref_mask=use_ref_mask)
    def generate(
        self,
        image=None, mask_image=None, prompt=None,
        pil_image=None, ref_mask=None, subject_text=None,
        num_samples=4, num_inference_steps=30,
        seed=None,
        image_file=None,
        negative_prompt=None, use_ref_mask=False,
        scale=1.0, guidance_scale=7.5,
        **kwargs,
    ):
        self.set_scale(scale)

        # [1, 3, 224, 224]
        ref_image_input = self.vis_processors["eval"](pil_image.convert("RGB")).unsqueeze(0)
        ref_text_input = self.txt_processors["eval"](subject_text)


        if use_ref_mask:
            # [1, 3, 768]
            sample = {"image": ref_image_input, "text_input": [ref_text_input], "raw_image": pil_image, "image_file": image_file.replace("/", "_").replace(".png", "")}
            multimodal_feature = self.blip_model.extract_features_with_ref_mask(sample, ref_mask, mode="multimodal", show_att=True).multimodal_embeds
        else:
            # [1, 3, 768]
            sample = {"image": ref_image_input, "text_input": [ref_text_input]}
            multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds

        multimodal_feature = torch.mean(multimodal_feature, dim=1)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = multimodal_feature.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, multimodal_feature=multimodal_feature
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)



        images = self.pipe(
            image=image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,

            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt_embeds=negative_prompt_embeds,
            **kwargs
        ).images

        return images

class anoSGAdapter_inpainting_fast:
    def __init__(self, sd_pipe, sg_ckpt, device, num_tokens=4, multimodal_name="blip", scale=1.0):
        self.device = device
        self.ip_ckpt = sg_ckpt
        self.num_tokens = num_tokens
        self.scale = scale

        self.pipe = sd_pipe.to(self.device)

        self.set_sg_adapter()

        # image proj model
        self.blip_proj_model, self.image_proj_model = self.init_proj()

        self.load_sg_adapter()

        # if multimodal_name == "blip":
        #     self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
        #         name="blip_feature_extractor",
        #         model_type="base", is_eval=True)
        # elif multimodal_name == "blip2":
        #     self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
        #         name="blip2_feature_extractor",
        #         model_type="pretrain", is_eval=True)



    def init_proj(self):
        blip_proj_model = BlipProjModel(
            input_embeddings_dim=768,
            output_attention_dim=1024).to(self.device, dtype=torch.float16)
        image_proj_model = ImageProjModel(
            # clip_embeddings_dim=768,
            clip_embeddings_dim=1024,  # 1024
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,  # 768
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)
        return blip_proj_model, image_proj_model

    def set_sg_adapter(self):
        unet = self.pipe.unet


        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SGAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=self.scale,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)


        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_sg_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "sg_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("sg_adapter."):
                        state_dict["sg_adapter"][key.replace("sg_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.blip_proj_model.load_state_dict(state_dict["blip_proj"])
        self.image_proj_model.load_state_dict(state_dict["image_proj"])


        sg_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        sg_layers.load_state_dict(state_dict["sg_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, multimodal_feature=None):
        multimodal_feature = multimodal_feature.to(self.device, dtype=torch.float16)

        blip_prompt_embeds = self.blip_proj_model(multimodal_feature)
        image_prompt_embeds = self.image_proj_model(blip_prompt_embeds)

        uncond_blip_prompt_embeds = self.blip_proj_model(torch.zeros_like(multimodal_feature))
        uncond_image_prompt_embeds = self.image_proj_model(uncond_blip_prompt_embeds)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SGAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        image=None, mask_image=None, prompt=None,
        pil_image=None, ref_mask=None, subject_text=None,
        pil_image_name=None, image_root_path=None,
        num_samples=4, num_inference_steps=30,
        seed=None,
        image_file=None,
        negative_prompt=None, use_ref_mask=False,
        scale=1.0, guidance_scale=7.5,
        **kwargs,
    ):
        self.set_scale(scale)


        self.cache_dir = os.path.join(image_root_path, "blip_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file = os.path.join(self.cache_dir, f"{pil_image_name.replace('/', '_').split('.')[0]}.pt")


        if os.path.exists(cache_file):
            multimodal_feature = torch.load(cache_file)
        else:
            # [1, 3, 224, 224]
            ref_image_input = self.vis_processors["eval"](pil_image.convert("RGB")).unsqueeze(0)
            ref_text_input = self.txt_processors["eval"](subject_text)



            if use_ref_mask:
                # [1, 3, 768]
                sample = {"image": ref_image_input, "text_input": [ref_text_input], "raw_image": pil_image,
                          "image_file": image_file.replace("/", "_").replace(".png", "")}
                multimodal_feature = self.blip_model.extract_features_with_ref_mask(sample, ref_mask, mode="multimodal",
                                                                                    show_att=True).multimodal_embeds
            else:
                # [1, 3, 768]
                sample = {"image": ref_image_input, "text_input": [ref_text_input]}
                multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds


            torch.save(multimodal_feature, cache_file)



        multimodal_feature = torch.mean(multimodal_feature, dim=1)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = multimodal_feature.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, multimodal_feature=multimodal_feature
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)



        images = self.pipe(
            image=image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,

            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt_embeds=negative_prompt_embeds,
            **kwargs
        ).images

        return images
