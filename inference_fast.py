import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt
from sg_adapter.ImageTextFusionAdapter import ImageTextFusionAdapter, anoSGAdapter, anoSGAdapter_inpainting, anoSGAdapter_inpainting_fast
import numpy as np
import json
import os
from scipy.ndimage import gaussian_filter

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def pil_to_numpy(image):
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    return image

def display_images(images, gap_size=10, main_title="", overview_save_path="generated_data_ref_mask/{dataset}_checkpoint_{save_steps}", image_id=""):


    images = [pil_to_numpy(image) for image in images]


    img_height, img_width, _ = images[0].shape


    new_width = img_width * len(images) + gap_size * 4
    new_image = np.ones((img_height, new_width, 3), dtype=np.uint8) * 255


    for i in range(len(images)):
        start_x = i * (img_width + gap_size)
        new_image[:, start_x:start_x + img_width, :] = images[i]

    plt.suptitle(main_title, fontsize=10)


    plt.imshow(new_image)
    plt.axis('off')
    # plt.show()


    save_path = os.path.join(overview_save_path, f"{image_id}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_models():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./pretrained_model/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    return pipe


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                      type=str,
                      required=True,
                      choices=["mvtec_ad", "mpdd", "visa"],
                      help="Specify the dataset to use (mvtec_ad/mpdd/visa)")
    args = parser.parse_args()
    
    sg_adapter_path = "./pretrained_model/"
    dataset = args.dataset
    print("Dataset used:", dataset)

    if dataset == "mvtec_ad":
        dataset_name = "MVTEC_AD"
    elif dataset == "mpdd":
        dataset_name = "MPDD"
    elif dataset == "visa":
        dataset_name = "VISA"

    multimodal_name = "blip2"
    use_ref_mask = True
    use_ano_ipadapter = True
    scale = 1.0
    save_dir = f"./save_model/{dataset}/ano_adapter_models_refmask={use_ref_mask}_blip={multimodal_name}_inpainting/"

    if dataset == "mpdd":
        save_steps = 200000
    elif dataset == "visa":
        save_steps = 200000
    else:
        save_steps = 200000
    print("Checkpoint used:", save_steps)
    ano_ip_ckpt = save_dir + f"checkpoint-{save_steps}/sg_adapter.bin"
    device = "cuda"

    pipe = load_models()


    inference_json_file_path = f'data/{dataset}/inference_data_dict.json'

    with open(inference_json_file_path, 'r') as f:
        data = json.load(f)


    for category, anomalies in data.items():
        for anomaly_type, items in anomalies.items():
            generated_count = 0
            for item in items:
                if generated_count >= 20:
                    break
                ori_image = item["ori_image"]
                ori_generated_mask = item["ori_generated_mask"]
                ref_image = item["ref_image_file"]
                ref_text = item["ref_text"]
                ref_mask = item["ref_mask_file"]
                image_id = f"{category}_{anomaly_type}_{item['image_id']}"

                image_root_folder = f"./data/{dataset}/{dataset_name}/"
                mask_root_folder = f"./data/{dataset}/generated_masks"

                image_file = os.path.join(image_root_folder, ori_image)
                mask_file = os.path.join(mask_root_folder, ori_generated_mask)
                ref_image_file = os.path.join(image_root_folder, ref_image)
                ref_mask_file = os.path.join(image_root_folder, ref_mask)

                image = Image.open(image_file).convert('RGB').resize((512, 512))
                mask = Image.open(mask_file).convert('L').resize((512, 512))
                ref_image = Image.open(ref_image_file).convert('RGB').resize((512, 512))
                ref_mask = Image.open(ref_mask_file).convert('L').resize((512, 512))

                num_samples = 1
                guidance_scale = 7

                sg_model = anoSGAdapter_inpainting_fast(pipe, ano_ip_ckpt, device, multimodal_name=multimodal_name,
                                                   scale=scale)

                images = sg_model.generate(image=image, mask_image=mask, prompt=ref_text,
                                           pil_image=ref_image, ref_mask=ref_mask, subject_text=ref_text,
                                           pil_image_name=item["ref_image_file"], image_root_path=image_root_folder,
                                           num_samples=num_samples, num_inference_steps=50,
                                           seed=42, strength=0.7,
                                           image_file=image_id, use_ref_mask=use_ref_mask,
                                           guidance_scale=guidance_scale)



                generated_images_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/image"
                if not os.path.exists(generated_images_save_path):
                    os.makedirs(generated_images_save_path)
                generated_images_files = os.path.join(generated_images_save_path, f"{image_id}.png")


                mask_array = np.array(mask, dtype=np.float32) / 255.0

                blurred_mask = gaussian_filter(mask_array, sigma=5)


                blurred_mask = (blurred_mask * 255).astype(np.uint8)
                blurred_mask_img = Image.fromarray(blurred_mask)


                composite_image = Image.composite(images[0], image, blurred_mask_img)
                composite_image.save(generated_images_files)

                generated_masks_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/mask"
                if not os.path.exists(generated_masks_save_path):
                    os.makedirs(generated_masks_save_path)
                generated_masks_files = os.path.join(generated_masks_save_path, f"{image_id}.png")
                mask.save(generated_masks_files)

                generated_ori_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/ori"
                if not os.path.exists(generated_ori_save_path):
                    os.makedirs(generated_ori_save_path)
                generated_ori_files = os.path.join(generated_ori_save_path, f"{image_id}.png")
                image.save(generated_ori_files)
                

                generated_ref_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/ref_image"
                if not os.path.exists(generated_ref_save_path):
                    os.makedirs(generated_ref_save_path)
                generated_ref_files = os.path.join(generated_ref_save_path, f"{image_id}.png")
                ref_image.save(generated_ref_files)
                

                generated_text_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/ref_text"
                if not os.path.exists(generated_text_save_path):
                    os.makedirs(generated_text_save_path)
                generated_text_files = os.path.join(generated_text_save_path, f"{image_id}.txt")
                with open(generated_text_files, 'w') as f:
                    f.write(ref_text)
                    

                generated_ref_mask_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/ref_mask"
                if not os.path.exists(generated_ref_mask_save_path):
                    os.makedirs(generated_ref_mask_save_path)
                generated_ref_mask_files = os.path.join(generated_ref_mask_save_path, f"{image_id}.png")
                ref_mask.save(generated_ref_mask_files)

                show_images = [image, mask, ref_image, ref_mask, composite_image]
                main_title = f"{image_id} \n {ref_text} \n use_ref_mask={use_ref_mask} \n scale={scale} \n blip_name={multimodal_name} \n {ref_text}"
                main_title += f" \n --from ano adapter inpainting checkpoint-{save_steps}" if use_ano_ipadapter else " \n --from original adapter"

                overview_save_path = f"generated_data_ref_mask/{dataset}_checkpoint_{save_steps}/{category}/{anomaly_type}/overview"
                if not os.path.exists(overview_save_path):
                    os.makedirs(overview_save_path)
                display_images(show_images, gap_size=10, main_title=main_title, overview_save_path=overview_save_path,
                               image_id=image_id)

                generated_count += 1
                print(f"{category} {anomaly_type} {image_id} (generated {generated_count}/20)")

if __name__ == "__main__":
    main()