# The first 1/3 of the smallest IDs of the object + anomaly categories are used for training, and the remaining 2/3 are used for testing.
# Save to ano_data_train.json and ano_data_test.json.
# ano_data_train.json can be used to train the anomaly synthesis model.

import os
import json

# Define the root path for the data folder
root = "../../"
root_folder = f"{root}data/mvtec_ad/MVTEC_AD/"

train_data = []
test_data = []
train_data_dict = {}
data = []

# Traverse all subfolders under the root folder as categories
for category in os.listdir(root_folder):
    
    if category == "blip_cache":
        continue
    category_path = os.path.join(root_folder, category)

    # Only process directories
    if os.path.isdir(category_path):

        train_data_dict[category] = {}

        # Traverse different damage types under each category
        test_folder = os.path.join(category_path, 'test')
        ground_truth_folder = os.path.join(category_path, 'ground_truth')
        prompt_folder = os.path.join(category_path, 'prompt')

        for broken_type in os.listdir(test_folder):

            if broken_type == "good":
                continue

            train_data_dict[category][broken_type] = []
            # Define paths
            test_broken_folder = os.path.join(test_folder, broken_type)
            ground_truth_broken_folder = os.path.join(ground_truth_folder, broken_type)
            prompt_broken_folder = os.path.join(prompt_folder, broken_type)

            # Get image files of the given type
            image_files = sorted([f for f in os.listdir(test_broken_folder) if f.endswith('.png')])

            split_index = len(image_files) // 3
            train_image_files = image_files[:split_index]
            test_image_files = image_files[split_index:]

            for image_file in train_image_files:
                image_path = os.path.join(test_broken_folder, image_file)
                mask_file = image_file.replace('.png', '_mask.png')
                mask_path = os.path.join(ground_truth_broken_folder, mask_file)

                # Get the corresponding text file and read the description
                txt_file = image_file.replace('.png', '.txt')
                txt_path = os.path.join(prompt_broken_folder, txt_file)

                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        text_description = f.read().strip()
                else:
                    text_description = "No description available."

                # Assume subject_text is the name of the damage type
                subject_text = broken_type.replace('_', ' ')

                # Use os.path.relpath to get the relative path
                entry = {
                    "image_file": os.path.relpath(image_path, root_folder),
                    "text": text_description,
                    "mask_file": os.path.relpath(mask_path, root_folder),
                    "subject_text": subject_text
                }
                train_data.append(entry)
                train_data_dict[category][broken_type].append(entry)

            for image_file in test_image_files:
                image_path = os.path.join(test_broken_folder, image_file)
                mask_file = image_file.replace('.png', '_mask.png')
                mask_path = os.path.join(ground_truth_broken_folder, mask_file)

                # Get the corresponding text file and read the description
                txt_file = image_file.replace('.png', '.txt')
                txt_path = os.path.join(prompt_broken_folder, txt_file)

                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        text_description = f.read().strip()
                else:
                    text_description = "No description available."

                # Assume subject_text is the name of the damage type
                subject_text = broken_type.replace('_', ' ')

                # Use os.path.relpath to get the relative path
                entry = {
                    "image_file": os.path.relpath(image_path, root_folder),
                    "text": text_description,
                    "mask_file": os.path.relpath(mask_path, root_folder),
                    "subject_text": subject_text
                }
                test_data.append(entry)

# Create the ../data folder (if it doesn't exist)
output_folder = "../../data/mvtec_ad"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the results as JSON files
train_output_file = os.path.join(output_folder, 'ano_data_train.json')
train_output_dict_file = os.path.join(output_folder, 'ano_data_dict_train.json')
test_output_file = os.path.join(output_folder, 'ano_data_test.json')

with open(train_output_file, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(test_output_file, 'w') as f:
    json.dump(test_data, f, indent=4)

with open(train_output_dict_file, 'w') as f:
    json.dump(train_data_dict, f, indent=4)

print(f"Training set JSON data has been generated and saved to {train_output_file}")
print(f"Test set JSON data has been generated and saved to {test_output_file}")

print(f"Training set JSON dictionary data has been generated and saved to {train_output_dict_file}")
