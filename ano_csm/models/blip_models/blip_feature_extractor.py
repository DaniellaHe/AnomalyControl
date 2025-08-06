"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings

import torch
import torch.nn.functional as F
from ano_csm.common.registry import registry
from ano_csm.models.blip_models.blip import BlipBase
from ano_csm.models.blip_models.blip_outputs import BlipOutputFeatures
from ano_csm.models.med import XBertEncoder
from ano_csm.models.vit import VisionTransformerEncoder
from torch import nn
from torchvision import transforms
# from visualizer import get_local
# get_local.activate()
import numpy as np
import matplotlib.pyplot as plt
import cv2

def gaussian(x, mu, sigma):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (torch.sqrt(2 * torch.tensor(3.14159265358979323846)) * sigma)
def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):


    # mask_np = masks[0].cpu().numpy()

    # plt.imshow(mask_np, cmap="gray", interpolation="nearest")
    # plt.title("Mask Visualization")

    # plt.show()

    loss = 0
    object_number = len(masks)
    # if object_number == 0:
    #     return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    if object_number == 0:
        return torch.zeros(1, requires_grad=True).cuda() if torch.cuda.is_available() else torch.zeros(1,requires_grad=True)

    attn_map = rel_map
    b = attn_map.shape[0]
    H, W = masks[0].shape

    for obj_idx in range(object_number):

        obj_loss = 0
        mask = masks[obj_idx]
        ca_map_obj = attn_map.reshape(b, H, W)

        if choice and choice in ["Scribble", "Point"]:
            activation_value = (ca_map_obj * gaussian(mask, 0, 0.1)).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b,
                                                                                                                     -1).sum(
                dim=-1)
        else:
            activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)

        # print(activation_value.grad_fn)

        obj_loss += torch.mean((1 - activation_value) ** 2)

        loss += obj_loss

    return loss


def extract_attention_map(module, input, output, attention_maps):

    if isinstance(output, tuple) and len(output) > 1:
        attention_probs = output[1]
        attention_maps.append(attention_probs)
    else:
        print("Attention map not found in output.")
def get_attention_maps(module, input, output, attention_maps):

    if len(output) > 1:

        if isinstance(output[1], (tuple, list)):
            total_logits = None
            for logits in output[1]:
                total_logits = logits if total_logits is None else total_logits + logits
            avg_logits = total_logits / len(output[1])

        elif isinstance(output[1], torch.Tensor):
            avg_logits = output[1]
        else:
            print(f"Unknown output[1] type: {type(output[1])}")
            return


        attention_weights = torch.softmax(avg_logits, dim=-1)


        attention_maps.append(attention_weights)

    else:
        print(f"Output length insufficient: {len(output)}")



def preprocess_mask(image, H=14, W=14, n_px=224):
    transform = transforms.Compose([
        transforms.Resize((n_px, n_px), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    return transform(image).squeeze(0)


def show_image_relevance(attention_map, image, preprocess, mask=None, show_mask=False,
                         att_hw=(24, 24), image_file=None, step=None):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask, alpha=0.5):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = alpha * heatmap + (1 - alpha) * np.float32(img)
        cam = cam / np.max(cam)
        return cam


    image_relevance = attention_map.detach().clone()

    plt.plot()
    fig = plt.gcf()
    # fig, axs = plt.subplots(1, 1)
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)

    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(56, 56), mode='bilinear')
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    image = preprocess(image)
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    # plt.imshow(image)
    plt.axis('off')
    if show_mask:
        # draw = ImageDraw.Draw(fig)
        mask = mask.reshape(1, 1, att_hw[0], att_hw[1])
        # mask = mask.reshape(1,1,16,16)
        mask = torch.nn.functional.interpolate(mask, size=224, mode='nearest')
        mask = mask.reshape(224, 224).cpu().numpy()
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(f'vis/{image_file}_mask.png', mask_image)

    fig.savefig(f'vis/{image_file}_{step}.png', dpi=300, bbox_inches='tight')

@registry.register_model("blip_feature_extractor")
class BlipFeatureExtractor(BlipBase):
    """
    Class for BLIP feature extractor.

    Supported model types:
        - base: BLIP base model with pre-trained weights from capfilt by BLIP large model.

    Usage:
        >>> from ano_csm.models import load_model
        >>> model = load_model("blip_feature_extractor", "base")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_feature_extractor_base.yaml",
        # "large": "configs/models/blip_feature_extractor_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim, max_txt_len=40):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.max_txt_len = max_txt_len

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.

        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".

        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from ano_csm.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel features

            image_embeds = self.visual_encoder.forward_features(image)



            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    # @torch.no_grad()
    def extract_features_with_ref_mask(self, samples, mask=None, mode="multimodal", show_att=False):
        image = samples.get("image")
        caption = samples.get("text_input")
        raw_image = samples.get("raw_image")
        image_file = samples.get("image_file")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":

            # return multimodel features

            image_embeds = self.visual_encoder.forward_features(image).detach()
            # image_embeds = self.visual_encoder.forward_features(image)


            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            mask_tensor = transforms.ToTensor()(mask)

            mask = preprocess_mask(mask, H=14, W=14, n_px=224)




            visual_prompt = torch.nn.Parameter(
                torch.zeros((1, image_embeds.size(1), image_embeds.size(2))).to(self.device), requires_grad=True)
            vprompt_history = visual_prompt


            optimizer = torch.optim.Adam([visual_prompt], lr=1e-1)
            num_steps = 3

            attention_maps = []
            for layer in self.text_encoder.encoder.layer:
                # print(layer)
                layer.crossattention.self.register_forward_hook(
                    lambda m, i, o: extract_attention_map(m, i, o, attention_maps))


            for step in range(num_steps):
                attention_maps = []

                optimizer.zero_grad()

                image_embeds_with_prompt = image_embeds + visual_prompt




                # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                #     self.device
                # )

                output = self.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=image_embeds_with_prompt,
                    # encoder_attention_mask=image_atts,
                    return_dict=True,
                    output_attentions=True
                )

                multimodal_embeds = output.last_hidden_state

                mean_attention_maps = torch.cat(attention_maps, 0).mean(0)
                fig = None
                if show_att:
                    H, W = 14, 14
                    n_px = 224
                    show_image_relevance(
                        mean_attention_maps[:, :, 1:].mean(axis=0).mean(axis=0),
                        raw_image,
                        mask=mask,
                        preprocess=transforms.Compose(
                            [transforms.Resize((n_px, n_px), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor()]),
                        show_mask=True,
                        att_hw=(H, W),
                        image_file=image_file,
                        step=step
                    )

                avg_attention_over_keys = mean_attention_maps[:, :, 1:].mean(axis=0).mean(axis=0).unsqueeze(0)
                alpha = 1
                loss = alpha * compute_ca_loss(avg_attention_over_keys, masks=[mask.to(avg_attention_over_keys.device)], object_positions=None)
                loss.backward()



                optimizer.step()
                # print(f"Step {step}, Loss: {loss.item()}")
            torch.cuda.empty_cache()


        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )
    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 30)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)
        else:
            warnings.warn("No pretrained weights are loaded.")

        return model
