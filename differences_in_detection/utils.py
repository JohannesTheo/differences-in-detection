import urllib.request
from pathlib import Path
from zipfile import ZipFile


def download_coco_annotations():

    annotation_dir = Path(__file__).parent.parent.resolve() / 'annotations'
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Download
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    output_file = (annotation_dir / "annotations_trainval2017.zip").resolve()

    if not output_file.exists():
        urllib.request.urlretrieve(url, output_file)
        print(f"Downloaded to: {annotation_dir}")

    # Extract
    if not (annotation_dir / 'instances_val2017.json').exists():
        with ZipFile(output_file, "r") as zip_ref:
            zip_ref.extractall(annotation_dir.parent)
        print(f"Extracted to: {annotation_dir}")
