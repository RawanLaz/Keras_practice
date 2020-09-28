import sys
import keras 

dataset_path = sys.argv[1]

# Create a dataset.
dataset = keras.preprocessing.image_dataset_from_directory(
  dataset_path, batch_size=64, image_size=(200, 200))

# For demonstration, iterate over the batches yielded by the dataset.
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
