# Generate `inference_data.json` from `ano_data_train.json`

# pill/train/good/aaa.png                                    ori_image
# pill/train/generated_mask/bbb.png                          ori_generated_mask
# "image_file": "pill/test/color/021.png",                   ref_image_file
# "text": "The pill has a red contamination.",               ref_text
# "mask_file": "pill/ground_truth/color/021_mask.png",       ref_mask_file

import os
import json
import random

# Set random seed
random.seed(42)

category_dict = {
    "carpet": {} ,
    "transistor": {} ,
    "cable": {} ,
    "zipper": {} ,
    "screw": {} ,
    "capsule": {} ,
    "wood": {} ,
    "tile": {} ,
    "grid": {} ,
    "toothbrush": {} ,
    "metal_nut": {} ,
    "hazelnut": {} ,
    "pill": {} ,
    "bottle": {} ,
    "leather": {} ,
}

def shuffle_and_get_top_300(input_list):
    # Create a copy of the list
    list_copy = input_list.copy()

    # Shuffle the list copy
    random.shuffle(list_copy)

    # Take the first 300 elements
    top_300_elements = list_copy[:300]

    return top_300_elements

def shuffle_and_get_top_2000(input_list):
    # Create a copy of the list
    list_copy = input_list.copy()

    # Shuffle the list copy
    random.shuffle(list_copy)

    # Take the first 2000 elements
    top_2000_elements = list_copy[:2000]

    return top_2000_elements

# Define the root path of the data folder
root = "../../"
image_root_folder = f"{root}data/mvtec_ad/MVTEC_AD/"
root_folder = f"{root}data/mvtec_ad/generated_masks"

ori_mask_data = []

# Iterate through all subfolders under root_folder as categories
for category in os.listdir(root_folder):
    if category == "blip_cache":
        continue
    category_path = os.path.join(root_folder, category)

    if '-' in category or '.' in category:
        continue

    print(category)

    ori_mask_data_tmp = []

    if os.path.isdir(category_path):

        for broken_type in os.listdir(category_path):

            print(category, broken_type)
            category_dict[category][broken_type] = []

            mask_folder = os.path.join(category_path, broken_type, 'mask')

            # Get mask files under this type
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])

            image_category_path = os.path.join(image_root_folder, category, 'train', 'good')
            image_files = sorted([f for f in os.listdir(image_category_path) if f.endswith('.png')])

            for mask_file in mask_files:
                mask_path = os.path.join(mask_folder, mask_file)

                for image_file in image_files:
                    img_category_path = os.path.join(image_root_folder, category)
                    img_folder = os.path.join(img_category_path, 'train', 'good')
                    image_path = os.path.join(img_folder, image_file)

                    # Use os.path.relpath to get relative path
                    ori_image = os.path.relpath(image_path, image_root_folder)

                    # Use os.path.relpath to get relative path
                    ori_generated_mask = os.path.relpath(mask_path, root_folder)

                    entry = {
                        "ori_image": ori_image,
                        "ori_generated_mask": ori_generated_mask
                    }

                    ori_mask_data_tmp.append(entry)

            ori_mask_data.extend(shuffle_and_get_top_2000(ori_mask_data_tmp))

            category_dict[category][broken_type].extend(shuffle_and_get_top_2000(ori_mask_data_tmp))

print(len(ori_mask_data))

# Create ../mvtec_ad if it does not exist
output_folder = f"../../data/mvtec_ad"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save results as JSON files
inference_data = os.path.join(output_folder, f'inference_data_{len(ori_mask_data)}.json')
inference_dict_data = os.path.join(output_folder, f'inference_dict_data_{len(ori_mask_data)}.json')

with open(inference_data, 'w') as f:
    json.dump(ori_mask_data, f, indent=4)

print("Saved to", inference_data)

with open(inference_dict_data, 'w') as f:
    json.dump(category_dict, f, indent=4)

print("Saved to", inference_dict_data)
