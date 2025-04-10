import fiftyone as fo
import fiftyone.zoo as foz

name = "my-dataset"
dataset_dir = "dataset/"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name=name,
)

session = fo.launch_app(dataset)
session.wait()  # Wait for the session to finish

