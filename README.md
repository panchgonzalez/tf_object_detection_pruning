# Pruning Mask R-CNN Models with TensorFlow


## Setup

Clone repo with submodules
```bash
git clone --recurse-submodules git@github.com:panchgonzalez/tf_object_detection_pruning.git
```

Apply patch to `tensorflow.contrib.model_pruning` library and compile

```bash
cd tensorflow

# Apply patch
git apply -v ../tf_model_pruning.patch

# Compile
bazel build tensorflow/contrib/model_pruning:strip_pruning_vars
```

Apply patch to `models.research.object_detection` and  `models.research.slim` libraries
that will allow us to prune InceptionV2 based MaskRCNN models

```bash
cd models

# Apply patch
git apply -v ../object_detection_pruning.patch

# Compile object detection protobufs
pushd research
protoc object_detection/protos/*.proto --python_out=.

# Install object detection
pip install .
popd

# Install slim
pushd research/slim
pip install .
popd

# Compiling and installing cocoapi
cd ../cocoapi
python setup.py build_ext --inplace
make
python setup.py install
cp -r pycocotools ../models/research
```


## References

Michael Zhu and Suyog Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression”, *2017 NIPS Workshop on Machine Learning of Phones and other Consumer Devices* (https://arxiv.org/pdf/1710.01878.pdf)
