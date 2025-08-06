import argparse
from pathlib import Path
import itertools
import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from sg_adapter.sg_adapter import ImageProjModel, BlipProjModel
from sg_adapter.utils import is_torch2_available

if is_torch2_available():
    from sg_adapter.attention_processor import SGAttnProcessor2_0 as SGAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from sg_adapter.attention_processor import SGAttnProcessor, AttnProcessor
from torch.utils.tensorboard import SummaryWriter
from datasets import *


class SGAdapter(torch.nn.Module):

    def __init__(self, unet, blip_proj_model, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.blip_proj_model = blip_proj_model
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        image_embeds = self.blip_proj_model(image_embeds)
        sg_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, sg_tokens], dim=1)
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            cross_attention_kwargs={"scale": 0.5}
        ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["sg_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--blip_name",
        type=str,
        default="blip2",
        help="",
    )
    parser.add_argument(
        "--use_mask",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="visa",
        choices=["mvtec_ad", "mpdd", "visa"],
        help="Select dataset to use (mvtec_ad/mpdd/visa)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="pretrained_model/stable-diffusion-inpainting/",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_sg_adapter_path",
        type=str,
        # default=None,
        default="pretrained_model/models/ip-adapter_sd15.bin",
        help="Path to pretrained ip adapter model.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=False,
        help="Training data json file",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="pretrained_model/models/image_encoder",
        required=False,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-sg_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():

    args = parse_args()

    print("Dataset used:", args.dataset)
    print(f"Saving checkpoint every {args.save_steps} steps")


    if args.data_root_path is None:
        if args.dataset == "mvtec_ad":
            args.data_root_path = "./data/mvtec_ad/MVTEC_AD/"
            args.data_json_file = "./data/mvtec_ad/ano_data_train.json"
        elif args.dataset == "mpdd":
            args.data_root_path = "./data/mpdd/MPDD"
            args.data_json_file = "./data/mpdd/ano_data_train.json"
        elif args.dataset == "visa":
            args.data_root_path = "./data/visa/VISA"
            args.data_json_file = "./data/visa/ano_data_train.json"

    
    args.output_dir = f"./save_model/{args.dataset}/ano_adapter_models_refmask={args.use_mask}_blip={args.blip_name}_inpainting"
    print("Model will be saved to:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with=args.report_to,
        project_config=accelerator_project_config,
        device_placement=True,
        gradient_accumulation_steps=2
    )


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )


    noise_scheduler = pipeline.scheduler
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # sg-adapter
    blip_proj_model = BlipProjModel(
        input_embeddings_dim=768,
        output_attention_dim=image_encoder.config.projection_dim)
    image_proj_model = ImageProjModel(
        # clip_embeddings_dim=768,
        clip_embeddings_dim=image_encoder.config.projection_dim,  # 1024
        cross_attention_dim=unet.config.cross_attention_dim,  # 768
        clip_extra_context_tokens=4,
    )

    print("image_encoder.config.projection_dim", image_encoder.config.projection_dim)
    print("unet.config.cross_attention_dim", unet.config.cross_attention_dim)

    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = SGAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    sg_adapter = SGAdapter(unet, blip_proj_model, image_proj_model, adapter_modules, args.pretrained_sg_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)


    params_to_opt = itertools.chain(
        sg_adapter.blip_proj_model.parameters(),
        sg_adapter.image_proj_model.parameters(),
        *(module.parameters() for module in sg_adapter.adapter_modules)
    )


    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-08
    )

    train_dataset = InpaintingDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution,
                                      image_root_path=args.data_root_path, use_mask=args.use_mask,
                                      blip_name=args.blip_name)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_inpainting,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    sg_adapter, optimizer, train_dataloader = accelerator.prepare(sg_adapter, optimizer, train_dataloader)

    global_step = 0

    writer = SummaryWriter(log_dir=args.logging_dir)
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        total_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            print("Epoch:", epoch, "Step:", step, "/", total_steps)
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(sg_adapter):
                with torch.no_grad():
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    multimodal_features = batch["multimodal_features"]

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(multimodal_features, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                    masked_latents = vae.encode(
                        batch["masked_images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor


                    mask = F.interpolate(batch["masks"], size=(args.resolution // 8, args.resolution // 8))

                latent_model_input = torch.cat([
                    noisy_latents,
                    mask,
                    masked_latents
                ], dim=1)

                noise_pred = sg_adapter(latent_model_input, timesteps, encoder_hidden_states, image_embeds)


                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {:.2f}s, time: {:.2f}s, step_loss: {:.4f}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss), flush=True)
                    writer.add_scalar("Loss/train", avg_loss, global_step)
                    if global_step % 10 == 0:
                        writer.flush()

            print(global_step)
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

                state_dict = {
                    "image_proj": sg_adapter.image_proj_model.state_dict(),
                    "sg_adapter": sg_adapter.adapter_modules.state_dict(),
                    "blip_proj": sg_adapter.blip_proj_model.state_dict()
                }
                torch.save(state_dict, os.path.join(save_path, "sg_adapter.bin"))

            global_step += 1

            begin = time.perf_counter()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print("Torch CUDA version:", torch.version.cuda)
    main()    
