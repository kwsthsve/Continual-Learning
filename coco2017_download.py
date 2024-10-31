from pycocotools.coco import COCO
import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def annotations_to_json(new_json, original_json, filenames):
    # Create dictionaries for new annotations
    new_annotations = {
        "images": [],
        "annotations": []
    }

    # Load original annotations
    with open(original_json, 'r') as annotation_file:
        original_annotations = json.load(annotation_file)

    # Filter annotations for the selected images
    for image_info in original_annotations['images']:
        if image_info['file_name'] in filenames:
            new_annotations['images'].append(image_info)
            annotations = [ann for ann in original_annotations['annotations'] if ann['image_id'] == image_info['id']]
            new_annotations['annotations'].extend(annotations)

    print("\nWriting annotations.json for images...\n")
    with open(new_json, "w") as json_file:
        json.dump(new_annotations, json_file, indent=4)


# Function that lets you view a cluster (based on identifier)
def view_cluster(cluster, dict, image_dir):
    plt.figure(figsize=(25, 25))

    # Gets the list of filenames for a cluster
    files = list(dict[cluster])

    print(f"Task {cluster} size = {len(files)}")

    # Only allow up to 30 images to be shown at a time
    if len(files) > 30:
        files = files[:29]

    # Plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = Image.open(image_dir + file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

    plt.show()


# Split to tasks using disjoint supercategories
def split2tasks_supercats(image_dir, new_json, caption_annFile, instance_annFile, path2txt, train=True):

    # Define five disjoint supercategories
    five_supercats = {
        'Living Things': ['person', 'animal'],
        'Vehicles and Outdoors': ['vehicle', 'outdoor', 'sports'],
        'Home and Furniture': ['furniture', 'indoor', 'appliance'],
        'Personal Items': ['accessory', 'electronic'],
        'Food and Kitchen': ['food', 'kitchen']
    }

    # Create a mapping from original supercategories to new ones
    original_to_new_supercat = {}
    for new_supercat, original_supercats in five_supercats.items():
        for original_supercat in original_supercats:
            original_to_new_supercat[original_supercat] = new_supercat

    # Load COCO dataset
    coco = COCO(instance_annFile)

    # Get all super-categories
    cats = coco.loadCats(coco.getCatIds())
    cat_to_supercat = {cat['id']: original_to_new_supercat[cat['supercategory']] for cat in cats}

    # Get all images and filter for our subset
    all_images = coco.loadImgs(coco.getImgIds())

    # Initialize clusters
    clusters = defaultdict(set)

    image_filenames = []

    # Cluster images by supercat (an image may be part of multiple supercategories, so an image belongs to more than one tasks)
    print('\n Clustering images to disjoint supercategories...')
    for img in tqdm(all_images):
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)

        supercats = set(cat_to_supercat[ann['category_id']] for ann in anns)

        # If image belongs to multiple new supercategories, put it in 'Other'
        if len(supercats) > 1:
            clusters['Other'].add(img['file_name'])

            # Check if the file exists before deleting
            if os.path.exists(image_dir + img['file_name']):
                os.remove(image_dir + img['file_name'])
        else:
            # If it belongs to exactly one category, add it to that category
            try:
                clusters[list(supercats)[0]].add(img['file_name'])
                image_filenames.append(img['file_name'])
            except IndexError:

                # Check if the file exists before deleting
                if os.path.exists(image_dir + img['file_name']):
                    os.remove(image_dir + img['file_name'])

                continue

    # Save the filenames selected
    with open(path2txt, 'w') as file:
        for filename in image_filenames:
            file.write(f"{filename}\n")

    # Create annotation file for selected images
    annotations_to_json(new_json, caption_annFile, image_filenames)

    # Print the distribution
    total_images = sum(len(images) for images in clusters.values())

    print(f"Total images: {total_images}")
    print("\nDistribution:")
    for supercat, images in clusters.items():
        print(f"{supercat}: {len(images)} images ({len(images) / total_images * 100:.2f}%)")

    # Calculate percentage of images in 'Other' category
    other_percentage = len(clusters['Other']) / total_images * 100 if 'Other' in clusters else 0
    print(f"\nPercentage of images in 'Other' category: {other_percentage:.2f}%")


    print('\nCreating directories and annotations for each task...')
    for cat in five_supercats.keys():

        # view_cluster(cat, five_supercats, image_dir)

        files = clusters[cat]

        if train:
            os.mkdir(f'./data/MSCOCO_annotations/task_{cat}_train_images')

            for img in files:
                image_data = Image.open(image_dir + img)
                image_data.save(f'./data/MSCOCO_annotations/task_{cat}_train_images/' + img)

            annotations_to_json(f'./data/MSCOCO_annotations/task_{cat}_train_annotations.json', new_json,
                                files)

        else:
            os.mkdir(f'./data/MSCOCO_annotations/task_{cat}_test_images')

            for img in files:
                image_data = Image.open(image_dir + img)
                image_data.save(f'./data/MSCOCO_annotations/task_{cat}_test_images/' + img)

            annotations_to_json(f'./data/MSCOCO_annotations/task_{cat}_test_annotations.json', new_json,
                                files)


# Training dataset arguments
train_cap_annFile = './data/MSCOCO_annotations/captions_train2017.json'
train_inst_annFile = './data/MSCOCO_annotations/instances_train2017.json'
train_image_dir = './data/MSCOCO_annotations/MSCOCO2017_train_images/'
new_train_json = './data/MSCOCO_annotations/filtered_train_annotations.json'
path2txt_train = './data/MSCOCO_annotations/train_image_filenames.txt'

split2tasks_supercats(train_image_dir, new_train_json, train_cap_annFile, train_inst_annFile, path2txt_train, train=True)


# Test dataset arguments
test_cap_annFile = './data/MSCOCO_annotations/captions_val2017.json'
test_inst_annFile = './data/MSCOCO_annotations/instances_val2017.json'
test_image_dir = './data/MSCOCO_annotations/MSCOCO2017_test_images/'
new_test_json = './data/MSCOCO_annotations/filtered_test_annotations.json'
path2txt_test = './data/MSCOCO_annotations/test_image_filenames.txt'

split2tasks_supercats(test_image_dir, new_test_json, test_cap_annFile, test_inst_annFile, path2txt_test, train=False)
