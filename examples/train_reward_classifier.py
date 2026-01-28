import sys
import glob
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from absl import app, flags

from serl_launcher_torch.data.data_store import ReplayBuffer
from serl_launcher_torch.utils.train_utils import concat_batches
from serl_launcher_torch.vision.data_augmentations import batched_random_crop
from serl_launcher_torch.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "cowa_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 9999, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
    # print(FLAGS.exp_name)
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, record_classifier_data = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=device,
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=device,
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    np.random.seed(0)

    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)
    
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    classifier, optimizer = create_classifier(
        sample=sample["observations"], 
        image_keys=config.classifier_keys,
        device=device
    )

    classifier.train()
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        
        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        
        # 数据增强 data_augmentation_fn
        # 注意：batched_random_crop 需要适配 PyTorch Tensors
        # 假设 batched_random_crop 可以处理字典输入并返回字典
        obs = batch["observations"]
        for pixel_key in config.classifier_keys:
            # 确保输入是 Tensor
            if isinstance(obs[pixel_key], torch.Tensor):
                # 假设 batched_random_crop 的 PyTorch 版本接受 Tensor 和 padding
                # 注意：PyTorch 中通常不需要 key，或者使用 torch.Generator
                obs[pixel_key] = batched_random_crop(obs[pixel_key], rng=rng, padding=4)
        
        # 更新 batch 中的 observations 和 labels
        # BCEWithLogitsLoss 需要 float 类型的标签
        labels = batch["labels"].float().unsqueeze(1) # Shape: [Batch, 1]
        
        # 将数据移动到设备 (虽然 ReplayBuffer 可能已经处理了，但确保一下)
        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
        labels = labels.to(device)

        # --- PyTorch 训练步骤 ---
        optimizer.zero_grad()
        
        # 前向传播
        logits = classifier(obs, train=True)
        
        # 计算损失
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        
        # 计算准确率
        preds = torch.sigmoid(logits) >= 0.5
        train_accuracy = (preds == labels).float().mean()
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()

        print(
            f"Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item():.4f}"
        )

    os.makedirs(os.path.join(os.getcwd(), "classifier_ckpt/"), exist_ok=True)
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.getcwd(), "classifier_ckpt/", "classifier.pth"))
    

if __name__ == "__main__":
    app.run(main)