import great_expectations as gx
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import os

context = gx.get_context()
print(f"Successfully created an EphemeralDataContext: {context}")

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="./data", download=True, train=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", download=True, train=False, transform=transform)

train_data = train_dataset.data
test_data = test_dataset.data

all_data = np.concatenate([train_data, test_data])
all_labels = np.concatenate([train_dataset.targets, test_dataset.targets])

cifar10_df = pd.DataFrame(all_data.reshape(all_data.shape[0], -1))
cifar10_df["label"] = all_labels

data_source = context.data_sources.add_pandas(name="cifar10_pandas")
data_asset = data_source.add_dataframe_asset(name="cifar10_asset")
data_batch_definition = data_asset.add_batch_definition_whole_dataframe(name="cifar10_batch_definition")
data_batch = data_batch_definition.get_batch(batch_parameters={"dataframe": cifar10_df})

print("Successfully added data asset.")
