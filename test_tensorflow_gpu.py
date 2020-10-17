# following this tutorial https://medium.com/analytics-vidhya/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a
import tensorflow as tf

# to check if Tensorflow has access to your GPU
print(tf.config.list_physical_devices('GPU'))
#ONLY to test CUDA support for your Tensorflow installation
print(tf.test.is_built_with_cuda())
#To confirm that the GPU is available to Tensorflow
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
#To check version
print(tf.__version__)
