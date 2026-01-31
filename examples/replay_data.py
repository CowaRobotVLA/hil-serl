import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from absl import app, flags
from PIL import Image
import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "cowa_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_string("data_file", "cowa_pick_1000_success_images_2026-01-22_20-07-40.pkl", "Path to the data file to replay.")
flags.DEFINE_string("output_dir", "./replay_images", "Directory to save extracted images.")
flags.DEFINE_boolean("save_video", True, "Whether to save images as a video.")
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

def extract_and_save_images(transitions, output_dir, save_video=True, fps=10):
    """从transitions中提取并保存图片"""
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = []
    print(f"\nExtracting images from {len(transitions)} transitions...")
    
    for i, transition in enumerate(tqdm(transitions, desc="Extracting images")):
        # 提取panorama/3图片
        if 'observations' in transition and 'panorama/3' in transition['observations']:
            img = transition['observations']['surround/front']
            
            # 确保图片格式正确
            if isinstance(img, np.ndarray):
                # 去掉batch维度：如果shape是(1, H, W, C)，转换为(H, W, C)
                if len(img.shape) == 4 and img.shape[0] == 1:
                    img = img[0]  # 去掉第一个维度
                
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
                
                # 保存单张图片
                # img_filename = os.path.join(output_dir, f"frame_{i:06d}.png")
                # Image.fromarray(img).save(img_filename)
                
                # 添加到视频列表
                images.append(img)
                
                # 每100张打印一次信息
                if i % 100 == 0 and i > 0:
                    print(f"\nProcessed {i} images, image shape: {img.shape}")
        else:
            print(f"\nWarning: 'panorama/3' not found in transition {i}")
    
    print(f"\nSaved {len(images)} images to {output_dir}")
    
    # 保存为视频
    if save_video and len(images) > 0:
        video_path = os.path.join(output_dir, "replay_video.mp4")
        print(f"\nCreating video at {video_path}...")
        
        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for img in tqdm(images, desc="Writing video"):
            # OpenCV使用BGR格式
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
        
        video_writer.release()
        print(f"Video saved to {video_path}")
    
    return images

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
    
    # 提取并保存图片
    images = extract_and_save_images(transitions, output_dir, FLAGS.save_video, FLAGS.fps)
    
    print(f"\nExtraction completed!")
    print(f"Images saved to: {output_dir}")
    if FLAGS.save_video:
        print(f"Video saved to: {os.path.join(output_dir, 'replay_video.mp4')}")

if __name__ == "__main__":
    app.run(main)