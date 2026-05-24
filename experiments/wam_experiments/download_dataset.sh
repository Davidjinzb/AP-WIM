# mkdir -p data/audiomnist && cd data/audiomnist

# wget -O train.tar.gz https://huggingface.co/datasets/flexthink/audiomnist/resolve/main/dataset/train.tar.gz
# wget -O valid.tar.gz https://huggingface.co/datasets/flexthink/audiomnist/resolve/main/dataset/valid.tar.gz
# wget -O test.tar.gz  https://huggingface.co/datasets/flexthink/audiomnist/resolve/main/dataset/test.tar.gz

# # 可选：元信息（说话人年龄/口音等）
# wget -O audioMNIST_meta.json https://huggingface.co/datasets/flexthink/audiomnist/resolve/main/meta/audioMNIST_meta.json


mkdir -p data/audiomnist/{train,valid,test}

tar -xf data/audiomnist/train.tar.gz -C data/audiomnist/train --strip-components=1
tar -xf data/audiomnist/valid.tar.gz -C data/audiomnist/valid --strip-components=1
tar -xf data/audiomnist/test.tar.gz  -C data/audiomnist/test  --strip-components=1