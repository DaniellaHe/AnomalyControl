# 使用 ano_data_dict_train.json + inference_dict_data_146000.json 生成. inference_data_dict.json（一共71100）

# pill/train/good/208.png                                   ori_image
# pill/contamination/mask/369.jpg                           ori_generated_mask
# "image_file": "pill/test/color/021.png",                  ref_image_file
# "text": "The pill has a red contamination.",              ref_text
# "mask_file": "pill/ground_truth/color/021_mask.png",      ref_mask_file

import json
import os
import random

# 设置随机种子
random.seed(42)

def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def shuffle_and_get_top_1000(input_list):
    # 创建列表副本
    list_copy = input_list.copy()

    # 打乱列表副本
    random.shuffle(list_copy)

    if len(list_copy) > 1000:
        # 取前1000个元素
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

        ref_images_masks= ano_data_train_type[broken_type]
        ori_images_masks = inference_data_type[broken_type]

        ori_ref_tmp = []

        # for ref in ref_images_masks:
        #     for ori in ori_images_masks:
        #         entry = {
        #             "ori_image": ori["ori_image"],
        #             "ori_generated_mask": ori["ori_generated_mask"],
        #             "ref_image_file": ref["image_file"],
        #             "ref_text": ref["text"],
        #             "ref_mask_file": ref["mask_file"],
        #         }
        #         ori_ref_tmp.append(entry)

        # 确保ref和ori的异常类型一致时才组合
        for ref in ref_images_masks:
            # 从ref文件名提取真实异常类型（如"poke"）
            ref_type = ref["mask_file"].split('/')[-2].replace('_mask', '')

            # 只选择同类型的ori数据
            matching_ori = [
                ori for ori in ori_images_masks
                if ref_type in ori["ori_generated_mask"]  # 检查路径是否包含类型关键字
            ]

            for ori in matching_ori:
                entry = {
                    "ori_image": ori["ori_image"],
                    "ori_generated_mask": ori["ori_generated_mask"],
                    "ref_image_file": ref["image_file"],
                    "ref_text": ref["text"].replace(ref_type, broken_type),  # 修正文本描述
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

print(f"推理数据已生成并保存到 {ori_ref_file}")

with open(ori_ref_dict_file, 'w') as f:
    json.dump(ori_ref_dict, f, indent=4)

print(f"推理数据已生成并保存到 {ori_ref_dict_file}")