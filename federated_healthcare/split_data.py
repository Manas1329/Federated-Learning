import os
import shutil
import random

# For reproducible results
random.seed(42)

# Dataset path
RAW_DATA_DIR = "chest_xray/train"

# Output folder
TARGET_BASE_DIR = "data"

# Number of images for each hospital
HOSPITAL_SPECS = {
    "hospital_A": {"NORMAL": 1000, "PNEUMONIA": 250},
    "hospital_B": {"NORMAL": 250, "PNEUMONIA": 1000},
    "hospital_C": {"NORMAL": 500, "PNEUMONIA": 500},
}


def create_non_iid_split():

    normal_src = os.path.join(RAW_DATA_DIR, "NORMAL")
    pneumonia_src = os.path.join(RAW_DATA_DIR, "PNEUMONIA")

    normal_files = os.listdir(normal_src)
    pneumonia_files = os.listdir(pneumonia_src)

    random.shuffle(normal_files)
    random.shuffle(pneumonia_files)

    normal_index = 0
    pneumonia_index = 0

    for hospital, counts in HOSPITAL_SPECS.items():

        normal_needed = counts["NORMAL"]
        pneumonia_needed = counts["PNEUMONIA"]

        hospital_normal = normal_files[
            normal_index:normal_index + normal_needed
        ]

        hospital_pneumonia = pneumonia_files[
            pneumonia_index:pneumonia_index + pneumonia_needed
        ]

        normal_index += normal_needed
        pneumonia_index += pneumonia_needed

        normal_target = os.path.join(
            TARGET_BASE_DIR,
            hospital,
            "train",
            "NORMAL"
        )

        pneumonia_target = os.path.join(
            TARGET_BASE_DIR,
            hospital,
            "train",
            "PNEUMONIA"
        )

        os.makedirs(normal_target, exist_ok=True)
        os.makedirs(pneumonia_target, exist_ok=True)

        for file in hospital_normal:
            shutil.copy(
                os.path.join(normal_src, file),
                os.path.join(normal_target, file)
            )

        for file in hospital_pneumonia:
            shutil.copy(
                os.path.join(pneumonia_src, file),
                os.path.join(pneumonia_target, file)
            )

        print(f"{hospital} created successfully.")

    print("\nDataset split completed!")


if __name__ == "__main__":
    create_non_iid_split()