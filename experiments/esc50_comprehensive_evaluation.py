#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESC-50 Comprehensive Evaluation Script

This script evaluates 6 explanation methods on the ESC-50 audio dataset:
1. ig (Integrated Gradients)
2. grad_cam (Gradient-weighted Class Activation Mapping)
3. saliency (Saliency Maps)
4. sg (SmoothGrad)
5. wam_ig (WAM with Integrated Gradients)
6. wam_sg (WAM with SmoothGrad)

Evaluation metrics:
1. ins (Insertion AUC)
2. del (Deletion AUC)
3. faith (Faithfulness = ins - del)
"""

import os
import json
import csv
import random
import traceback
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 项目内接口
from src.helpers import load_audio_model
from src.evaluators import EvalAudioBaselines, Eval1DWAM
from src.dataloader import ESC50

# -----------------------
# Config
# -----------------------
ESC50_ROOT = "./ESC-50-master"
MODEL_PATH = "./test_models/best_model_fold1.pt"

N_SAMPLES = 100                 # 抽取多少个样本做平均
N_ITER = 128                   # insertion/deletion 步数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "./outputs_esc50"
os.makedirs(OUT_DIR, exist_ok=True)

# 复现实验
random.seed(42)

# -----------------------
# Data Loading
# -----------------------
class ESC50AudioDataset:
    """ESC-50数据集类，返回原始音频波形"""
    
    def __init__(self, data_dir: str, fold: int = 1, train: bool = False, sample_rate: int = 44100, duration: float = 5.0):
        """
        Args:
            data_dir: ESC-50数据集目录路径
            fold: 测试折数 (1-5)
            train: 是否为训练模式
            sample_rate: 目标采样率
            duration: 音频时长（秒）
        """
        import pandas as pd
        import torchaudio
        import torchaudio.transforms as T
        
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # 加载元数据
        metadata_file = os.path.join(data_dir, 'meta', 'esc50.csv')
        self.metadata = pd.read_csv(metadata_file)
        
        # 根据fold过滤数据
        if train:
            self.data = self.metadata[self.metadata['fold'] != fold].reset_index(drop=True)
        else:
            self.data = self.metadata[self.metadata['fold'] == fold].reset_index(drop=True)
        
        print(f"ESC50AudioDataset: {'Training' if train else 'Testing'} set with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """加载并处理音频文件"""
        import torchaudio
        import torchaudio.transforms as T
        
        row = self.data.iloc[idx]
        filename = row['filename']
        target = row['target']
        
        # 加载音频文件
        audio_path = os.path.join(self.audio_dir, filename)
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 填充或裁剪到目标长度
            if waveform.shape[1] > self.target_length:
                # 从中心裁剪
                start = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            elif waveform.shape[1] < self.target_length:
                # 用零填充
                pad_length = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            return waveform, int(target)
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # 返回零填充的波形
            waveform = torch.zeros((1, self.target_length))
            return waveform, int(target)


def load_esc50_data(fold: int = 1, max_samples: int = None, data_dir: str = None):
    """加载ESC-50数据集
    
    Args:
        fold: 测试折数 (1-5)
        max_samples: 最大样本数量，用于测试
        data_dir: ESC-50数据集目录路径
    
    Returns:
        list: [(audio_waveform, label), ...] 其中audio_waveform是1D张量
    """
    if data_dir is None:
        data_dir = './ESC-50-master'
    
    # 创建数据集实例
    dataset = ESC50AudioDataset(data_dir=data_dir, fold=fold, train=False)
    
    # 获取数据
    data_list = []
    for i in range(len(dataset)):
        if max_samples and i >= max_samples:
            break
        
        audio_waveform, label = dataset[i]
        data_list.append((audio_waveform, label))
        
    print(f"Loaded {len(data_list)} samples from ESC-50 fold {fold}")
    if len(data_list) > 0:
        print(f"Audio waveform shape: {data_list[0][0].shape}")
    
    return data_list

# -----------------------
# Model Wrapper
# -----------------------
class ModelWrapper(nn.Module):
    """包装模型以确保只返回主要输出"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]  # 只返回主要输出
        return output

# -----------------------
# Evaluation Functions
# -----------------------
def _nan_safe(v):
    """安全处理NaN值"""
    return float(v) if np.isfinite(v) else float('nan')

def get_target_layer(model, method_name):
    """获取模型的目标层用于GradCAM"""
    if method_name == 'grad_cam':
        # 为weak_mxh64_1024模型选择合适的层
        original_model = model.model if hasattr(model, 'model') else model
        if hasattr(original_model, 'layer18'):  # weak_mxh64_1024 - 最后一个卷积层
            return original_model.layer18
        elif hasattr(original_model, 'layer16'):  # weak_mxh64_1024 - 倒数第二个卷积层
            return original_model.layer16  
        elif hasattr(original_model, 'layer'):  # 其他模型
            return original_model.layer
        else:
            # 寻找最后一个卷积层
            last_conv = None
            for module in original_model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            return last_conv
    return None

def run_explanation_method(method_name: str, model, audio_waveform, y_idx: int, device: str, n_iter: int = 128):
    """运行指定的解释方法评估
    
    Args:
        method_name: 方法名称 ('ig', 'grad_cam', 'saliency', 'sg', 'wam_ig', 'wam_sg')
        model: 模型
        audio_waveform: 1D音频波形数据 [batch_size, waveform_length]
        y_idx: 目标类别索引
        device: 设备
        n_iter: 迭代次数
    
    Returns:
        tuple: (insertion_auc, deletion_auc, faithfulness)
    """
    # 确保输入在正确设备上
    audio_waveform = audio_waveform.to(device)
    y = torch.tensor([y_idx], dtype=torch.long, device=device)
    
    try:
        # 映射方法名称到内部实现
        method_mapping = {
            'ig': 'integratedgrad',
            'grad_cam': 'gradcam', 
            'saliency': 'saliency',
            'sg': 'smoothgrad',
            'wam_ig': 'wam_integratedgrad',
            'wam_sg': 'wam_smooth'
        }
        
        internal_method = method_mapping.get(method_name)
        if not internal_method:
            raise ValueError(f"Unknown method: {method_name}")
        
        if method_name in ['ig', 'grad_cam', 'saliency', 'sg']:
            # 基线方法 - 使用EvalAudioBaselines
            original_model = model.model if hasattr(model, 'model') else model
            evaluator = EvalAudioBaselines(
                method_name=internal_method, 
                model=original_model, 
                batch_size=8, 
                device=device
            )
            
            # 为GradCAM添加layer属性
            if method_name == 'grad_cam':
                target_layer = get_target_layer(original_model, method_name)
                if target_layer is None:
                    print(f"    Cannot find target layer for GradCAM")
                    return float('nan'), float('nan'), float('nan')
                # 临时添加layer属性
                original_model.layer = target_layer
            
            # 计算insertion和deletion分数
            ins_scores = evaluator.insertion(audio_waveform, y, target="melspec", n_iter=n_iter)
            del_scores = evaluator.deletion(audio_waveform, y, target="melspec", n_iter=n_iter)
            
        elif method_name in ['wam_ig', 'wam_sg']:
            # WAM方法需要1D音频输入
            original_model = model.model if hasattr(model, 'model') else model
            wam_method = internal_method.replace('wam_', '')  # 'integratedgrad' or 'smooth'
            
            # 创建WAM评估器
            evaluator = Eval1DWAM(
                model=original_model,
                method=wam_method,
                batch_size=8,
                device=device,
                n_mels=128,
                n_fft=1024,
                sample_rate=44100
            )
            
            # 计算insertion和deletion分数
            ins_result = evaluator.insertion(audio_waveform, y, target="melspec", n_iter=n_iter)
            del_result = evaluator.deletion(audio_waveform, y, target="melspec", n_iter=n_iter)
            
            # 提取分数（第一个元素是分数列表）
            ins_scores = ins_result if isinstance(ins_result, list) else ins_result[0] if isinstance(ins_result, tuple) else [ins_result]
            del_scores = del_result if isinstance(del_result, list) else del_result[0] if isinstance(del_result, tuple) else [del_result]
            
        else:
            raise ValueError(f"Unknown method category: {method_name}")
        
        # 计算faithfulness = insertion - deletion
        ins_auc = float(ins_scores[0]) if len(ins_scores) > 0 else float('nan')
        del_auc = float(del_scores[0]) if len(del_scores) > 0 else float('nan')
        faith_score = ins_auc - del_auc if (np.isfinite(ins_auc) and np.isfinite(del_auc)) else float('nan')
        
        return _nan_safe(ins_auc), _nan_safe(del_auc), _nan_safe(faith_score)
        
    except Exception as e:
        print(f"    Method {method_name} failed with error: {e}")
        return float('nan'), float('nan'), float('nan')

# -----------------------
# Main Evaluation
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='ESC-50 Comprehensive Evaluation')
    parser.add_argument('--data_dir', type=str, default=ESC50_ROOT, help='ESC-50 dataset directory')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--fold', type=int, default=1, help='Test fold number (1-5)')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use (cuda/cpu)')
    parser.add_argument('--n_iter', type=int, default=N_ITER, help='Number of iterations for insertion/deletion')
    parser.add_argument('--output_dir', type=str, default=OUT_DIR, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=N_SAMPLES, help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = load_audio_model(args.model_path, device=device)
    
    # 包装模型以处理返回元组的问题
    model = ModelWrapper(model)
    model.eval()
    print(f"Model loaded successfully: {type(model)}")
    
    # 加载数据
    print(f"Loading ESC-50 data from: {args.data_dir}")
    data_list = load_esc50_data(args.fold, args.max_samples, args.data_dir)
    print(f"Dataset loaded, fold {args.fold}")
    
    # 定义要评估的6种方法
    methods = ['ig', 'grad_cam', 'saliency', 'sg', 'wam_ig', 'wam_sg']
    
    # 初始化结果存储
    per_sample_results: List[Dict[str, Any]] = []
    agg = {m: {'ins': [], 'del': [], 'faith': []} for m in methods}
    
    # 主评估循环
    sample_count = 0
    for sample_idx, (audio_waveform, y_idx) in enumerate(data_list):
        if sample_count >= args.max_samples:
            break
            
        # 确保数据在正确设备上
        audio_waveform = audio_waveform.to(device)  # [1, waveform_length]
        
        filename = f"sample_{sample_idx}"
        
        print(f"\n[{sample_count + 1}/{args.max_samples}] Processing: {filename}")
        print(f"  Input shape: {audio_waveform.shape}, Target class: {y_idx}")
        
        # 记录当前样本信息
        rec = {
            'sample_index': sample_count,
            'filename': filename,
            'target_class_index': y_idx,
            'input_shape': list(audio_waveform.shape),
            'scores': {}
        }
        
        try:
            # 测试所有6种方法
            for method in methods:
                print(f"  Running method: {method}")
                try:
                    ins, del_score, faith = run_explanation_method(
                        method, model, audio_waveform, y_idx, str(device), n_iter=args.n_iter
                    )
                    
                    rec['scores'][method] = {
                        'ins': ins,
                        'del': del_score,
                        'faith': faith
                    }
                    
                    # 添加到聚合结果
                    agg[method]['ins'].append(ins)
                    agg[method]['del'].append(del_score)
                    agg[method]['faith'].append(faith)
                    
                    print(f"    {method}: ins={ins:.4f}, del={del_score:.4f}, faith={faith:.4f}")
                    
                except Exception as e:
                    print(f"    {method} failed: {e}")
                    rec['scores'][method] = {
                        'ins': float('nan'),
                        'del': float('nan'),
                        'faith': float('nan')
                    }
                    agg[method]['ins'].append(float('nan'))
                    agg[method]['del'].append(float('nan'))
                    agg[method]['faith'].append(float('nan'))
                    
        except Exception as e:
            rec['error'] = f"{type(e).__name__}: {e}"
            rec['traceback'] = traceback.format_exc()
            print(f"  Sample failed with error: {e}")
        
        per_sample_results.append(rec)
        sample_count += 1
        
        print(f"[{sample_count}/{args.max_samples}] Done: {filename}")
    
    # 计算平均结果
    def _mean_or_nan(vals):
        finite_vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite_vals)) if len(finite_vals) > 0 else float('nan')
    
    mean_results = {}
    for method in methods:
        mean_results[method] = {
            'ins_mean': _mean_or_nan(agg[method]['ins']),
            'del_mean': _mean_or_nan(agg[method]['del']),
            'faith_mean': _mean_or_nan(agg[method]['faith']),
            'valid_samples': len([v for v in agg[method]['ins'] if np.isfinite(v)])
        }
    
    # 打印结果
    print(f"\n=== ESC-50 Fold {args.fold} Comprehensive Evaluation Results (n={sample_count}) ===")
    print(f"{'Method':<10} | {'ins':<8} | {'del':<8} | {'faith':<8} | {'valid':<5}")
    print("-" * 50)
    for method, results in mean_results.items():
        print(f"{method:<10} | {results['ins_mean']:<8.4f} | "
              f"{results['del_mean']:<8.4f} | "
              f"{results['faith_mean']:<8.4f} | "
              f"{results['valid_samples']:<5}")
    
    # 保存结果
    output_data = {
        'config': {
            'data_dir': args.data_dir,
            'model_path': args.model_path,
            'fold': args.fold,
            'device': str(device),
            'n_iter': args.n_iter,
            'max_samples': args.max_samples,
            'actual_samples': sample_count,
            'methods': methods
        },
        'per_sample_results': per_sample_results,
        'mean_results': mean_results
    }
    
    # JSON输出
    out_json = os.path.join(args.output_dir, f"esc50_comprehensive_fold{args.fold}_results.json")
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved JSON: {out_json}")
    
    # CSV输出
    out_csv = os.path.join(args.output_dir, f"esc50_comprehensive_fold{args.fold}_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ins_mean", "del_mean", "faith_mean", "valid_samples"])
        for method, results in mean_results.items():
            writer.writerow([
                method,
                results["ins_mean"],
                results["del_mean"],
                results["faith_mean"],
                results["valid_samples"]
            ])
    print(f"Saved CSV: {out_csv}")

if __name__ == "__main__":
    main()