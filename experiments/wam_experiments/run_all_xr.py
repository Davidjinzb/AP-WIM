import os
import multiprocessing
attr_methods = [
    "wam_mfaba",
]
datasets = [
    "imagenet",
    # "esc50",
    # "esc50"
    # "cub"
]
models = [
    "efficient_b0",
    # "resnet50",
    # "resnet18"
    # "mobilenet_v2"
]
steps = [1,3,5,7,10,13,15,17,20]
pool = multiprocessing.Pool(processes=4)
cuda_flag = False
commands = []
# for model in models:
#     for dataset in datasets:
for model,dataset in zip(models,datasets):
        for attr_method in attr_methods:
            for step in steps:
                # cmd_attribution = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python generate_attribution_xr.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug --steps {step}"
                # cmd_eval = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python eval_xr.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug --steps {step}"
                cmd_road = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python test_road_xr.py --model {model} --dataset {dataset} --attr_method {attr_method} --debug --steps {step} --baseline noise"
                # command = f"{cmd_attribution} && {cmd_eval} && {cmd_road}"
                command = f"{cmd_road}"
                commands.append(command)
                cuda_flag = not cuda_flag  # Alternate between using and not using CUDA

pool.imap(os.system, commands)
pool.close()
pool.join()
