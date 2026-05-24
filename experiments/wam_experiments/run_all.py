import os
import multiprocessing
attr_methods = [
    # "wam_ig",
    # "wam_sg",
    "wam_mfaba",
    # "la",
    # "agi",
    # "ig",
    # "fast_ig",
    # "sg",
    # "saliencymap",
    # "mfaba"
]
datasets = [
    "imagenet",
    # "esc50"
    # "cub"
]
models = [
    "efficient_b0",
    # "resnet50"
    # "resnet18"
    # "mobilenet_v2"
]
pool = multiprocessing.Pool(processes=4)
cuda_flag = False
commands = []
# for model in models:
#     for dataset in datasets:
for model,dataset in zip(models,datasets):
        for attr_method in attr_methods:
            cmd_attribution = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python generate_attribution.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug"
            cmd_eval = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python eval.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug"
            cmd_road = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python test_road.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug"
            command = f"{cmd_attribution} && {cmd_eval} && {cmd_road}"
            commands.append(command)
            cuda_flag = not cuda_flag  # Alternate between using and not using CUDA

pool.imap(os.system, commands)
pool.close()
pool.join()
