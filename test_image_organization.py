import os
import shutil
import re


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

SOURCE_DIR = os.path.join(
    DATA_DIR,
    "Test_All_Photos"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "Test"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXTRACT SAMPLE ID
# ============================================================

def find_sample_id(filename):
    """
    Extracts the test soil sample name from the filename.

    Example:

    iPhone14_HPC_Audorfring (1).JPG
        -> Audorfring

    iPhone14_HPC_Münster_BS6_9_0-10m (1).JPG
        -> Münster_BS6_9_0-10m

    iPhone16_HPC_Airbus BS10-4bis7 (2).JPG
        -> Airbus BS10-4bis7

    iPhone16_HPC_Airbus BS6-3 (2).JPG
        -> Airbus BS6-3

    iPhone14_HPC_Testfeld Lidl WHV (1).JPG
        -> Testfeld Lidl WHV
    """

    # Remove extension
    name = os.path.splitext(filename)[0]

    # Remove phone + HPC prefix
    name = re.sub(
        r"^iPhone(?:14|16)_HPC_",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove image number at the end
    # Examples:
    # "(1)"
    # "(2)"
    # "(10)"
    name = re.sub(
        r"\s*\(\d+\)$",
        "",
        name
    )

    return name.strip()


# ============================================================
# FIND TEST IMAGES
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
    file
    for file in files
    if os.path.splitext(file)[1] in image_extensions
]


print(f"Found {len(image_files)} test images.")


# ============================================================
# ORGANIZE IMAGES
# ============================================================

organized_count = 0

sample_counts = {}


for filename in sorted(image_files):

    sample_id = find_sample_id(filename)

    if not sample_id:
        print(
            f"[WARNING] Could not identify sample: {filename}"
        )
        continue

    # Create sample folder
    sample_dir = os.path.join(
        OUTPUT_DIR,
        sample_id
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    # Source image
    source_path = os.path.join(
        SOURCE_DIR,
        filename
    )

    # Destination image
    destination_path = os.path.join(
        sample_dir,
        filename
    )

    # Copy image
    shutil.copy2(
        source_path,
        destination_path
    )

    # Count
    sample_counts[sample_id] = (
        sample_counts.get(sample_id, 0) + 1
    )

    organized_count += 1

    print(
        f"[OK] {filename}  -->  {sample_id}/"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEST IMAGE ORGANIZATION COMPLETE")
print("=" * 60)

print(
    f"Total test images : {len(image_files)}"
)

print(
    f"Images organized  : {organized_count}"
)

print(
    f"Test samples      : {len(sample_counts)}"
)

print("\nImages per test sample:")

for sample_id, count in sorted(sample_counts.items()):

    print(
        f"  {sample_id}: {count} images"
    )