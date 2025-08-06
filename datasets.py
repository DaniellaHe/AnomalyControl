import os
import random
import json
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor

import sys
import os


local_lavis_path = "ano_csm"
sys.path.insert(0, local_lavis_path)

from ano_csm.models import load_model_and_preprocess

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 image_root_path="", use_mask=False, blip_name="blip"):
        super().__init__()

        self.cache_dir = os.path.join(image_root_path, "blip_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.use_mask = use_mask
        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

        if blip_name == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor",
                                                                              model_type="base", is_eval=True)
        elif blip_name == "blip2":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain", is_eval=True)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]
        subject_text = item["subject_text"]
        mask_file = item["mask_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        # [1, 3, 224, 224]
        # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # read mask
        mask = Image.open(os.path.join(self.image_root_path, mask_file)).convert("L")
        # mask = self.transform(mask)

        # [1, 3, 224, 224]
        image_input = self.vis_processors["eval"](raw_image.convert("RGB")).unsqueeze(0)
        subject_text_input = self.txt_processors["eval"](text)


        cache_file = os.path.join(self.cache_dir, f"{image_file.replace('/', '_')}.pt")
        

        if os.path.exists(cache_file):
            multimodal_feature = torch.load(cache_file)
        else:

            if self.use_mask:
                sample = {"image": image_input, "text_input": [subject_text_input], "raw_image": raw_image,
                          "image_file": image_file.replace("/", "_").replace(".png", "")}
                multimodal_feature = self.blip_model.extract_features_with_ref_mask(
                    sample, mask, mode="multimodal", show_att=False).multimodal_embeds
            else:
                sample = {"image": image_input, "text_input": [subject_text_input]}
                multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds
            

            torch.save(multimodal_feature, cache_file)



        with torch.cuda.amp.autocast(dtype=torch.float16):
            multimodal_feature = torch.mean(multimodal_feature, dim=1)
            multimodal_feature = multimodal_feature.detach().to(torch.float16)
        

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            # "clip_image": clip_image,
            "multimodal_feature": multimodal_feature,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)

class InferDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 image_root_path="", use_mask=False, blip_name="blip"):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.use_mask = use_mask
        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

        if blip_name == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor",
                                                                              model_type="base", is_eval=True)
        elif blip_name == "blip2":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain", is_eval=True)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]
        subject_text = item["subject_text"]
        mask_file = item["mask_file"]

        # read image
        image = Image.open(os.path.join(self.image_root_path, image_file))
        # read mask
        mask = Image.open(os.path.join(self.image_root_path, mask_file)).convert("L")
        # mask = self.transform(mask)
        # [1, 3, 224, 224]
        image_input = self.vis_processors["eval"](image.convert("RGB")).unsqueeze(0)
        subject_text_input = self.txt_processors["eval"](subject_text)

        if self.use_mask:
            # [1, 3, 768]
            sample = {"image": image_input, "text_input": [subject_text_input], "raw_image": image,
                      "image_file": image_file.replace("/", "_").replace(".png", "")}
            multimodal_feature = self.blip_model.extract_features_with_ref_mask(sample, mask, mode="multimodal",
                                                                                show_att=False).multimodal_embeds
        else:
            sample = {"image": image_input, "text_input": [subject_text_input]}
            # [1, 3, 768]
            multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds

        multimodal_feature = torch.mean(multimodal_feature, dim=1)
        multimodal_feature = multimodal_feature.detach()

        return {
            "image": image,
            "text": text,
            "multimodal_feature": multimodal_feature,
            "mask": mask,
            "subject_text": subject_text,
            "image_id": image_file,
        }

    def __len__(self):
        return len(self.data)

class InpaintingDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 image_root_path="", use_mask=False, blip_name="blip"):
        super().__init__()

        self.cache_dir = os.path.join(image_root_path, "blip_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.use_mask = use_mask
        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

        if blip_name == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip_feature_extractor",
                model_type="base", is_eval=True)
        elif blip_name == "blip2":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain", is_eval=True)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]
        subject_text = item["subject_text"]
        mask_file = item["mask_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))  
        # [1, 3, 224, 224]
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values  

        # read mask
        mask = Image.open(os.path.join(self.image_root_path, mask_file)).convert("L")
        tensor_mask = self.transform(mask)  


        masked_image = image * (tensor_mask < 0.5)

        # [1, 3, 224, 224]
        image_input = self.vis_processors["eval"](raw_image.convert("RGB")).unsqueeze(0)  
        subject_text_input = self.txt_processors["eval"](text)


        cache_file = os.path.join(self.cache_dir, f"{image_file.replace('/', '_').split('.')[0]}.pt")


        if os.path.exists(cache_file):
            multimodal_feature = torch.load(cache_file)
        else:

            if self.use_mask:
                sample = {"image": image_input, "text_input": [subject_text_input], "raw_image": raw_image,
                          "image_file": image_file.replace("/", "_").replace(".png", "")}
                multimodal_feature = self.blip_model.extract_features_with_ref_mask(
                    sample, mask, mode="multimodal", show_att=False).multimodal_embeds
            else:
                sample = {"image": image_input, "text_input": [subject_text_input]}
                multimodal_feature = self.blip_model.extract_features(sample, mode="multimodal").multimodal_embeds


            torch.save(multimodal_feature, cache_file)

        multimodal_feature = torch.mean(multimodal_feature, dim=1)
        multimodal_feature = multimodal_feature.detach()
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "images": image,
            "masks": tensor_mask,
            "masked_images": masked_image,
            "text_input_ids": text_input_ids,
            "multimodal_feature": multimodal_feature,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)


def pil_collate_fn(batch):
    return batch

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    multimodal_features = torch.stack([example["multimodal_feature"] for example in data]).to(torch.float32)
    drop_image_embeds = torch.tensor([example["drop_image_embed"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "multimodal_features": multimodal_features,
        "drop_image_embeds": drop_image_embeds
    }

def collate_fn_inpainting(data):
    images = torch.stack([example["images"] for example in data])
    masks = torch.stack([example["masks"] for example in data])
    masked_images = torch.stack([example["masked_images"] for example in data])

    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)

    multimodal_features = torch.cat([example["multimodal_feature"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    return {
        "images": images,
        "masks": masks,
        "masked_images": masked_images,
        "text_input_ids": text_input_ids,
        "multimodal_features": multimodal_features,
        "drop_image_embeds": drop_image_embeds
    }