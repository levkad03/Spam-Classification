from datasets import load_dataset

ds = load_dataset("Deysi/spam-detection-dataset")
ds.save_to_disk("spam-detection-dataset")
