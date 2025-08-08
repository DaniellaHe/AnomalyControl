# 使用 ano_data_train.json 生成 inference_data_7300.json

# pill/train/good/aaa.png                                    ori_image
# pill/train/generated_mask/bbb.png                          ori_generated_mask
# "image_file": "pill/test/color/021.png",                  ref_image_file
# "text": "The pill has a red contamination.",              ref_text
# "mask_file": "pill/ground_truth/color/021_mask.png",      ref_mask_file

# 全选有924万，太多了。
# 每个物体+破损类别随机选了300张。有21900张。
import os
import json
import random

# 设置随机种子
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
    # 创建列表副本
    list_copy = input_list.copy()

    # 打乱列表副本
    random.shuffle(list_copy)

    # 取前300个元素
    top_300_elements = list_copy[:300]

    return top_300_elements

def shuffle_and_get_top_2000(input_list):
    # 创建列表副本
    list_copy = input_list.copy()

    # 打乱列表副本
    random.shuffle(list_copy)

    # 取前2000个元素
    top_2000_elements = list_copy[:2000]

    return top_2000_elements

# 定义数据文件夹根路径
root = "../../"
root = "/apdcephfs/default101636/apdcephfs_qy3/share_1372663/shidanhe/code/AnomalyControl_anydev/"
image_root_folder = f"{root}data/mvtec_ad/MVTEC_AD/"
root_folder = f"{root}data/mvtec_ad/generated_masks"

ori_mask_data = []

# 遍历root文件夹下的所有子文件夹作为类别
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

            # 获取该类型下的图片文件
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])



            image_category_path = os.path.join(image_root_folder, category, 'train', 'good')
            image_files = sorted([f for f in os.listdir(image_category_path) if f.endswith('.png')])

            for mask_file in mask_files:
                mask_path = os.path.join(mask_folder, mask_file)

                for image_file in image_files:
                    img_category_path = os.path.join(image_root_folder, category)
                    img_folder = os.path.join(img_category_path, 'train', 'good')
                    image_path = os.path.join(img_folder, image_file)

                    # 使用os.path.relpath获取相对路径
                    ori_image = os.path.relpath(image_path, image_root_folder)

                    # 使用os.path.relpath获取相对路径
                    ori_generated_mask = os.path.relpath(mask_path, root_folder)

                    entry = {
                        "ori_image": ori_image,
                        "ori_generated_mask": ori_generated_mask
                    }

                    ori_mask_data_tmp.append(entry)

            ori_mask_data.extend(shuffle_and_get_top_2000(ori_mask_data_tmp))

            category_dict[category][broken_type].extend(shuffle_and_get_top_2000(ori_mask_data_tmp))
            # print(len(category_dict[category]))

print(len(ori_mask_data))

# 创建../mvtec_ad（如果不存在）
output_folder = f"../../data/mvtec_ad"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 将结果保存为JSON文件
inference_data = os.path.join(output_folder, f'inference_data_{len(ori_mask_data)}.json')
inference_dict_data = os.path.join(output_folder, f'inference_dict_data_{len(ori_mask_data)}.json')

with open(inference_data, 'w') as f:
    json.dump(ori_mask_data, f, indent=4)

print("已经保存到", inference_data)

with open(inference_dict_data, 'w') as f:
    json.dump(category_dict, f, indent=4)

print("已经保存到", inference_dict_data)


# entry = {
#     "ref_image_file": os.path.relpath(image_path, root_folder),
#     "ref_text": text_description,
#     "ref_mask_file": os.path.relpath(mask_path, root_folder),
# }