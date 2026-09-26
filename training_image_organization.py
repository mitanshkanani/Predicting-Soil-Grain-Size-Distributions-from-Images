import os
import shutil
import pandas as pd
import re


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

SOURCE_DIR = os.path.join(
    DATA_DIR,
    "Training-All_Photos_updated"
)

LABEL_FILE = os.path.join(
    DATA_DIR,
    "Training_labels_updated.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "Training"
)


# ============================================================
# LOAD LABELS
# ============================================================

labels_df = pd.read_csv(LABEL_FILE)

sample_ids = labels_df["sample_id"].astype(str).str.strip().tolist()

print(f"Found {len(sample_ids)} labeled soil samples.")

# Create a lookup dictionary
labels_lookup = {
    str(row["sample_id"]).strip(): row
    for _, row in labels_df.iterrows()
}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for sample_id in sample_ids:
    sample_dir = os.path.join(
        OUTPUT_DIR,
        sample_id
    )

    os.makedirs(sample_dir, exist_ok=True)


# ============================================================
# FIND SAMPLE ID FROM FILENAME
# ============================================================

def find_sample_id(filename):
    """
    Finds the soil sample ID from a training image filename.

    Examples:
        Motorola_Edge_F827_01.jpg
        Motorola_Edge_G190_02.jpg
        Motorola_Edge_H030_04.jpg
        Motorola_Edge_60_fusion_H374_01.jpg

    Returns:
        Sample ID such as F827, G190, H030, H374
    """

    for sample_id in sample_ids:

        # Look for the sample ID as a separate filename component.
        pattern = rf"(^|_){re.escape(sample_id)}(_|\.|$)"

        if re.search(pattern, filename, re.IGNORECASE):
            return sample_id

    return None


# ============================================================
# ORGANIZE IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

files = os.listdir(SOURCE_DIR)

image_files = [
    file for file in files
    if os.path.splitext(file)[1] in image_extensions
]

print(f"Found {len(image_files)} training images.")

organized_count = 0
unmatched_files = []


for filename in sorted(image_files):

    sample_id = find_sample_id(filename)

    if sample_id is None:

        unmatched_files.append(filename)

        print(
            f"[WARNING] Could not identify sample ID: {filename}"
        )

        continue

    source_path = os.path.join(
        SOURCE_DIR,
        filename
    )

    destination_dir = os.path.join(
        OUTPUT_DIR,
        sample_id
    )

    destination_path = os.path.join(
        destination_dir,
        filename
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    organized_count += 1

    print(
        f"[OK] {filename}  -->  {sample_id}/"
    )


# ============================================================
# SAVE LABELS INSIDE EACH SAMPLE FOLDER
# ============================================================

target_columns = [
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200"
]


for sample_id in sample_ids:

    row = labels_lookup[sample_id]

    sample_label = {
        "sample_id": sample_id
    }

    for column in target_columns:
        sample_label[column] = row[column]

    label_df = pd.DataFrame([sample_label])

    label_path = os.path.join(
        OUTPUT_DIR,
        sample_id,
        "labels.csv"
    )

    label_df.to_csv(
        label_path,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ORGANIZATION COMPLETE")
print("=" * 60)

print(f"Total labeled soil samples : {len(sample_ids)}")
print(f"Total images found         : {len(image_files)}")
print(f"Images organized            : {organized_count}")
print(f"Unmatched images            : {len(unmatched_files)}")


if unmatched_files:

    print("\nUnmatched files:")

    for filename in unmatched_files:
        print(f"  - {filename}")


print("\nImages per soil sample:")

for sample_id in sample_ids:

    sample_dir = os.path.join(
        OUTPUT_DIR,
        sample_id
    )

    count = sum(
        1
        for file in os.listdir(sample_dir)
        if os.path.splitext(file)[1] in image_extensions
    )

    print(f"  {sample_id}: {count} images")