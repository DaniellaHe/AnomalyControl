# Generate `inference_data_dict.json` (total 71100) from `ano_data_dict_train.json` + `inference_dict_data_146000.json`

# pill/train/good/208.png                                   ori_image
# pill/contamination/mask/369.jpg                           ori_generated_mask
# "image_file": "pill/test/color/021.png",                   ref_image_file
# "text": "The pill has a red contamination.",               ref_text
# "mask_file": "pill/ground_truth/color/021_mask.png",       ref_mask_file

import json
import os
import random

# Set random seed
random.seed(42)

def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def shuffle_and_get_top_1000(input_list):
    # Create a copy of the list
    list_copy = input_list.copy()

    # Shuffle the list copy
    random.shuffle(list_copy)

    if len(list_copy) > 1000:
        # Take the first 1000 elements
        top_1000_elements = list_copy[:1000]
    else:
        top_1000_elements = list_copy

    return top_1000_elements

def add_image_id(data_list):
    for idx, item in enumerate(data_list, start=1):
        item['image_id'] = idx
    return data_list

ano_data_train = "../../data/mvtec_ad/ano_data_dict_train.json"
inference_data_146000 = "../../data/mvtec_ad/inference_dict_data_146000.json"

ano_data_train_dict = read_json(ano_data_train)
inference_data_146000_dict = read_json(inference_data_146000)

ori_ref = []
ori_ref_dict = {}
categories = ano_data_train_dict.keys()
for category in categories:
    ano_data_train_type = ano_data_train_dict[category]
    inference_data_type = inference_data_146000_dict[category]
    broken_types = ano_data_train_type.keys()
    ori_ref_dict[category] = {}
    for broken_type in broken_types:
        print(category, broken_type)

        ori_ref_dict[category][broken_type] = []

        ref_images_masks = ano_data_train_type[broken_type]
        ori_images_masks = inference_data_type[broken_type]

        ori_ref_tmp = []

        # Ensure ref and ori have the same anomaly type before combining
        for ref in ref_images_masks:
            # Extract the actual anomaly type from the ref filename (e.g., "poke")
            ref_type = ref["mask_file"].split('/')[-2].replace('_mask', '')

            # Select only ori data with the same type
            matching_ori = [
                ori for ori in ori_images_masks
                if ref_type in ori["ori_generated_mask"]  # Check if the path contains the type keyword
            ]

            for ori in matching_ori:
                entry = {
                    "ori_image": ori["ori_image"],
                    "ori_generated_mask": ori["ori_generated_mask"],
                    "ref_image_file": ref["image_file"],
                    "ref_text": ref["text"].replace(ref_type, broken_type),  # Adjust the text description
                    "ref_mask_file": ref["mask_file"],
                }
                ori_ref_tmp.append(entry)
        print(len(ori_ref_tmp))
        top_1000_elements = shuffle_and_get_top_1000(ori_ref_tmp)
        top_1000_elements = add_image_id(top_1000_elements)
        ori_ref.extend(top_1000_elements)
        ori_ref_dict[category][broken_type].extend(top_1000_elements)

print(len(ori_ref))

output_folder = "../../data/mvtec_ad"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ori_ref_file = os.path.join(output_folder, 'inference_data.json')
ori_ref_dict_file = os.path.join(output_folder, 'inference_data_dict.json')

with open(ori_ref_file, 'w') as f:
    json.dump(ori_ref, f, indent=4)

print(f"Inference data has been generated and saved to {ori_ref_file}")

with open(ori_ref_dict_file, 'w') as f:
    json.dump(ori_ref_dict, f, indent=4)

print(f"Inference data has been generated and saved to {ori_ref_dict_file}")
