import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from absl import app, flags
from PIL import Image
import cv2
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "cowa_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_string("data_file", "cowa_pick_10_success_images_2026-01-31_12-19-00.pkl", "Path to the data file to replay.")
flags.DEFINE_string("output_dir", "./replay_images", "Directory to save extracted images.")
flags.DEFINE_integer("fps", 10, "Frames per second for video.")
flags.DEFINE_integer("max_transitions", -1, "Maximum number of transitions to process (-1 for all).")

def list_available_files(exp_name):
    """列出可用的数据文件"""
    data_dir = "./classifier_data"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []
    
    files = [f for f in os.listdir(data_dir) if f.startswith(exp_name) and f.endswith('.pkl')]
    return sorted(files)

def load_data(file_path):
    """加载pkl数据文件"""
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_data(data, file_path):
    """保存pkl数据文件"""
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)
    print(f"Data saved to: {file_path}")

def display_and_filter_transitions(transitions, output_dir):
    """交互式显示并筛选transitions"""
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 用于保存单张图片的固定文件名
    temp_image_path = os.path.join(output_dir, "current_frame.png")
    
    kept_transitions = []
    kept_images = []
    removed_count = 0
    
    print(f"\n开始交互式筛选，共 {len(transitions)} 个样本")
    print("=" * 60)
    print("操作说明:")
    print("  输入 's' 或 'S' - 保留当前数据")
    print("  输入 'r' 或 'R' - 删除当前数据")
    print("  输入 'q' 或 'Q' - 退出筛选")
    print("=" * 60)
    
    for i, transition in enumerate(transitions):
        # 提取panorama/3图片
        if 'observations' in transition and 'panorama/3' in transition['observations']:
            img = transition['observations']['surround/front']
            
            # 确保图片格式正确
            if isinstance(img, np.ndarray):
                # 去掉batch维度：如果shape是(1, H, W, C)，转换为(H, W, C)
                if len(img.shape) == 4 and img.shape[0] == 1:
                    img = img[0]
                
                # 如果是浮点数，转换到0-255范围
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # 如果是单通道，转换为RGB
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # 保存到固定文件（覆盖上一张）
                Image.fromarray(img).save(temp_image_path)
                
                # 显示当前进度
                print(f"\n[{i+1}/{len(transitions)}] 当前图片已保存到: {temp_image_path}")
                print(f"已保留: {len(kept_transitions)} | 已删除: {removed_count}")
                
                # 等待用户输入
                while True:
                    user_input = input("请选择操作 (s=保留, r=删除, q=退出): ").strip().lower()
                    
                    if user_input == 's':
                        kept_transitions.append(transition)
                        kept_images.append(img.copy())  # 保存图片副本
                        print("✓ 已保留")
                        break
                    elif user_input == 'r':
                        removed_count += 1
                        print("✗ 已删除")
                        break
                    elif user_input == 'q':
                        print("\n提前退出筛选...")
                        return kept_transitions, kept_images, removed_count, True
                    else:
                        print("无效输入，请输入 s、r 或 q")
        else:
            print(f"\nWarning: 'panorama/3' not found in transition {i}, skipping...")
    
    # 删除临时图片文件
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    return kept_transitions, kept_images, removed_count, False

def create_video_from_images(images, output_path, fps=10):
    """从图片列表创建视频"""
    if len(images) == 0:
        print("没有图片可以生成视频")
        return
    
    print(f"\n创建视频: {output_path}")
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in tqdm(images, desc="生成视频"):
        # OpenCV使用BGR格式
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    
    video_writer.release()
    print(f"✓ 视频已保存: {output_path}")

def main(_):
    # 如果没有指定文件，列出所有可用文件
    if not FLAGS.data_file:
        print(f"\nAvailable data files for {FLAGS.exp_name}:")
        files = list_available_files(FLAGS.exp_name)
        if not files:
            print("No data files found.")
            return
        
        for idx, f in enumerate(files):
            print(f"{idx}: {f}")
        
        print("\nPlease specify a file using --data_file=<filename>")
        print("Or use the full path: --data_file=./classifier_data/<filename>")
        return
    
    # 加载数据
    data_path = FLAGS.data_file
    if not os.path.exists(data_path):
        # 尝试在classifier_data目录中查找
        data_path = os.path.join("./classifier_data", FLAGS.data_file)
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {FLAGS.data_file}")
        return
    
    print(f"Loading data from: {data_path}")
    transitions = load_data(data_path)
    print(f"Loaded {len(transitions)} transitions")
    
    # 限制处理数量
    if FLAGS.max_transitions > 0:
        transitions = transitions[:FLAGS.max_transitions]
        print(f"Will process first {len(transitions)} transitions")
    
    # 创建输出目录名（基于输入文件名）
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(FLAGS.output_dir, base_name)
    
    # 交互式筛选
    kept_transitions, kept_images, removed_count, early_exit = display_and_filter_transitions(
        transitions, output_dir
    )
    
    # 显示筛选结果
    print("\n" + "=" * 60)
    print("筛选完成!")
    print("=" * 60)
    print(f"原始数据: {len(transitions)} 个样本")
    print(f"保留数据: {len(kept_transitions)} 个样本")
    print(f"删除数据: {removed_count} 个样本")
    print("=" * 60)
    
    if len(kept_transitions) == 0:
        print("\n没有保留任何数据，不生成输出文件")
        return
    
    # 保存筛选后的PKL文件
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filtered_pkl_name = f"{base_name}_filtered_{timestamp}.pkl"
    filtered_pkl_path = os.path.join(output_dir, filtered_pkl_name)
    save_data(kept_transitions, filtered_pkl_path)
    
    # 生成视频
    video_name = f"{base_name}_filtered_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_name)
    create_video_from_images(kept_images, video_path, FLAGS.fps)
    
    print("\n" + "=" * 60)
    print("所有文件已生成:")
    print(f"  PKL文件: {filtered_pkl_path}")
    print(f"  MP4视频: {video_path}")
    print("=" * 60)

if __name__ == "__main__":
    app.run(main)