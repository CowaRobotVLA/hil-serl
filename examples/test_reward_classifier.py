import copy
import os
from tqdm import tqdm
import numpy as np
from absl import app, flags
from datetime import datetime
from PIL import Image

from experiments.mappings import CONFIG_MAPPING
from experiments.cowa_pick.config import TrainConfig, EnvConfig
from serl_robot_infra.robot_env.envs.relative_env import RelativeFrame
from serl_launcher_torch.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher_torch.wrappers.chunking import ChunkingWrapper
from serl_launcher_torch.networks.reward_classifier import load_classifier_func
from serl_robot_infra.robot_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
import time
from experiments.cowa_pick.wrapper import PickEnv, GripperPenaltyWrapper

class TestConig(TrainConfig):
    def get_environment(self, fake_env=False, save_video=False, classifier=False, device = "cuda"):
        env = PickEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                sample=env.observation_space.sample(),
                image_keys=self.image_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/classifier.pth"),
                device=device,
            )

            # def reward_func(obs):
            #     sigmoid = lambda x: 1 / (1 + torch.exp(-x))
            #     return int(sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4)
            def reward_func(obs):
                return int(classifier(obs) > 0.5)

            env = TestRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env

class TestRewardClassifierWrapper(MultiCameraBinaryRewardClassifierWrapper):
    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "cowa_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("samples_needed", 500, "Number of samples to collect for analysis.")
flags.DEFINE_string("fp_save_dir", "false_positives", "Directory to save false positive images.")

success_key = []

def save_observation_images(obs, save_dir, sample_id, image_keys):
    """保存观测中的所有图片"""
    os.makedirs(save_dir, exist_ok=True)
    
    for key in image_keys:
        if key in obs and obs[key] is not None:
            # 获取图片数据
            img_data = obs[key]
            
            # 如果是numpy数组，转换为PIL Image
            if isinstance(img_data, np.ndarray):
                # 处理不同的形状: (H, W, C) 或 (1, H, W, C)
                if img_data.ndim == 4:
                    img_data = img_data[0]  # 取第一帧
                
                # 确保数据在0-255范围内
                if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                    img_data = (img_data * 255).astype(np.uint8)
                
                # 转换为PIL Image
                img = Image.fromarray(img_data)
                
                # 将key中的 '/' 替换为 '_'，避免被当作路径分隔符
                safe_key = key.replace('/', '_').replace('\\', '_')
                
                # 保存图片
                img_filename = f"sample_{sample_id:05d}_{safe_key}.png"
                img_path = os.path.join(save_dir, img_filename)
                img.save(img_path)


def calculate_accuracy_metrics(results):
    """计算分类器的各项准确率指标"""
    true_positives = sum(1 for r in results if r['ground_truth'] == True and r['predicted'] == 1)
    true_negatives = sum(1 for r in results if r['ground_truth'] == False and r['predicted'] == 0)
    false_positives = sum(1 for r in results if r['ground_truth'] == False and r['predicted'] == 1)
    false_negatives = sum(1 for r in results if r['ground_truth'] == True and r['predicted'] == 0)
    
    total = len(results)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total
    }

def print_metrics(metrics):
    """打印分类器性能指标"""
    print("\n" + "="*60)
    print("分类器准确率分析结果")
    print("="*60)
    print(f"总样本数: {metrics['total_samples']}")
    print(f"\n准确率 (Accuracy): {metrics['accuracy']:.2%}")
    print(f"精确率 (Precision): {metrics['precision']:.2%}")
    print(f"召回率 (Recall): {metrics['recall']:.2%}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  真阳性 (TP): {metrics['true_positives']}")
    print(f"  真阴性 (TN): {metrics['true_negatives']}")
    print(f"  假阳性 (FP): {metrics['false_positives']}")
    print(f"  假阴性 (FN): {metrics['false_negatives']}")
    print("="*60 + "\n")

def main(_):
    global success_key
    
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = TestConig()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    success_key = env.success_key
    
    # 创建保存假阳性图片的目录（添加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp_save_dir = os.path.join(FLAGS.fp_save_dir, timestamp)
    os.makedirs(fp_save_dir, exist_ok=True)
    print(f"假阳性图片将保存到: {fp_save_dir}")
    
    obs, _ = env.reset()
    
    # 只保留必要的数据: ground_truth (success/failure) 和 predicted (reward 0/1)
    results = []
    samples_needed = FLAGS.samples_needed
    pbar = tqdm(total=samples_needed, desc="Collecting samples")
    
    fp_count = 0  # 统计假阳性数量
    
    while len(results) < samples_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # 判断是否为假阳性
        ground_truth = success_key[0]
        predicted = int(rew)
        is_false_positive = (ground_truth == False and predicted == 1)
        
        # 如果是假阳性，保存图片
        if is_false_positive:
            fp_count += 1
            sample_id = len(results)
            save_observation_images(next_obs, fp_save_dir, sample_id, config.image_keys)
            print(f"\n检测到假阳性 #{fp_count}，已保存图片 (sample_{sample_id:05d})")
        
        # 只保留ground truth标签和分类器预测结果
        result = {
            'ground_truth': ground_truth,  # True for success, False for failure
            'predicted': predicted  # 分类器输出的0或1
        }
        results.append(result)
        pbar.update(1)
        
        # 每100个样本显示一次中间结果
        if len(results) % 100 == 0:
            temp_metrics = calculate_accuracy_metrics(results)
            pbar.set_postfix({
                'Accuracy': f"{temp_metrics['accuracy']:.2%}",
                'FP': fp_count
            })
        
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
    
    pbar.close()
    
    # 计算并打印最终指标
    final_metrics = calculate_accuracy_metrics(results)
    print_metrics(final_metrics)
    print(f"\n总共检测到 {fp_count} 个假阳性样本")
    print(f"假阳性图片已保存到: {fp_save_dir}")

if __name__ == "__main__":
    app.run(main)