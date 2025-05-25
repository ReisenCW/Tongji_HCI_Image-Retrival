import upstash_vector as uv
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
import os
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import gradio as gr

# 初始化配置
class Config:
    upstash_url = "https://worthy-marmoset-72553-us1-vector.upstash.io"
    token = "ABkFMHdvcnRoeS1tYXJtb3NldC03MjU1My11czFhZG1pbk1HRm1abVkwTldNdFptRTBOQzAwT0dFekxXRTRNekF0TURFeFpHSXpZVE5qTXpCaw=="
    dataset_root = "./GroceryStoreDataset/dataset/"  # 图片根目录
    load_txt_path = "./GroceryStoreDataset/dataset/train.txt"
    batch_size = 64 if torch.cuda.is_available() else 8
    upload_batch = 128
    num_workers = 0 if os.name == 'nt' else 4
    top_k = 5 # 返回图片数

# 初始化Upstash
index = uv.Index(url=Config.upstash_url, token=Config.token)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # 先移动设备，再调整精度
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

if device.type == 'cuda':
    model = model.half()  # GPU使用半精度
else:
    model = model.float()  # CPU使用单精度
model = model.eval()

# 图像预处理流水线
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

class ImageDataset(Dataset):
    def __init__(self):
        with open(Config.load_txt_path, 'r') as f:
            lines = f.readlines()

        self.image_paths = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 1:
                continue

            # 组合完整路径
            rel_path = parts[0].strip()
            abs_path = os.path.join(Config.dataset_root, rel_path)
            if os.path.isfile(abs_path):
                self.image_paths.append(abs_path)
            else:
                logger.warning(f"Missing image: {abs_path}")

        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx not in self.cache:
            try:
                with Image.open(self.image_paths[idx]) as img:
                    img = img.convert("RGB")
                    self.cache[idx] = preprocess(img)
            except Exception as e:
                logger.error(f"Error loading {self.image_paths[idx]}: {str(e)}")
                return None, None
        return self.cache[idx], self.image_paths[idx]


# 批处理函数
@torch.no_grad()
def batch_embed(images_batch):
    images_batch = images_batch.to(device, non_blocking=True)
    if device.type == 'cuda':
        images_batch = images_batch.half()  # 显式转换为半精度
    with torch.autocast(device_type='cuda', dtype=torch.float16 if device.type == 'cuda' else torch.float32):
        return model.get_image_features(pixel_values=images_batch)


# 文本查询函数
@torch.no_grad()
def query_text(text_query, top_k=Config.top_k):
    # 文本编码
    inputs = tokenizer(text_query, return_tensors="pt", padding=True).to(device)

    float_keys = ['attention_mask']  # 只转换需要浮点类型的键
    inputs = {
        k: v.half() if device.type == 'cuda' and k in float_keys else v
        for k, v in inputs.items()
    }

    # 生成文本特征
    text_features = model.get_text_features(**inputs)
    query_vector = text_features.squeeze().cpu().numpy().astype(
        np.float32).tolist()

    # 执行查询
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 返回图片路径和分数
    return [(res.metadata["path"], float(res.score)) for res in results if
            res.metadata]
# 异步上传执行器
upload_executor = ThreadPoolExecutor(max_workers=2)


def async_upload(batch_vectors):
    def upload_task():
        try:
            index.upsert(vectors=batch_vectors)
            logger.info(f"Successfully uploaded {len(batch_vectors)} vectors")
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            # 失败重试逻辑
            for _ in range(3):
                try:
                    index.upsert(vectors=batch_vectors)
                    logger.info("Retry upload succeeded")
                    return
                except:
                    continue
            logger.error("Failed after 3 retries")

    upload_executor.submit(upload_task)

def custom_collate(batch):
    valid_items = [item for item in batch if item  [0] is not None and item  [1] is not None]
    if not valid_items:
        return None, None
    images = torch.stack([item  [0] for item in valid_items])
    paths = [item  [1] for item in valid_items]
    return images, paths

# 主处理流程
def process_images():
    dataset = ImageDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )

    vectors_buffer = []
    for batch in dataloader:
        if batch is None or batch[0] is None:
            continue
        images, paths = batch

        # 批量推理
        embeddings = batch_embed(images).cpu().numpy().astype(np.float32)

        # 构造向量数据
        batch_vectors = [
            (p, emb.tolist(), {"path": p})  # 使用完整路径作为ID
            for p, emb in zip(paths, embeddings)
        ]

        vectors_buffer.extend(batch_vectors)

        # 批量上传
        if len(vectors_buffer) >= Config.upload_batch:
            async_upload(vectors_buffer[:Config.upload_batch])
            vectors_buffer = vectors_buffer[Config.upload_batch:]

    # 上传剩余数据
    if vectors_buffer:
        async_upload(vectors_buffer)
    upload_executor.shutdown(wait=True)


# 查询函数
@torch.no_grad()
def query_image(image_path, top_k=Config.top_k):
    # 预处理
    with Image.open(image_path) as img:
        img = preprocess(img.convert("RGB")).unsqueeze(0).to(device).half()

    # 推理
    embedding = model.get_image_features(pixel_values=img)
    query_vector = embedding.squeeze().cpu().numpy().astype(np.float32).tolist()

    # 查询
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return [(res.metadata["path"], float(res.score)) for res in results]


# 创建Gradio界面
def create_interface():
    dataset_cache = ImageDataset()
    def search_text(query, top_k=Config.top_k):
        results = query_text(query, top_k)
        return load_result_images(results)

    def search_image(img, top_k=Config.top_k):
        if img is None or not os.path.exists(img):
            return []
        results = query_image(img, top_k)
        return load_result_images(results)

    def get_random_samples(n=3):
        if len(dataset_cache.image_paths) == 0:
            return []
        indices = np.random.choice(
            len(dataset_cache.image_paths),
            size=min(n, len(dataset_cache.image_paths)),
            replace=False
        )
        return [dataset_cache.image_paths[i] for i in indices]

    def refresh_samples():
        samples = get_random_samples(3)
        return [s if os.path.exists(s) else None for s in samples]

    # 通用结果加载函数
    def load_result_images(results):
        images = []
        for path, score in results:
            if os.path.exists(path):
                try:
                    images.append(Image.open(path))
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
            else:
                logger.error(f"Path {path} does not exist.")
        return images

    # 标签点击回调函数
    def update_text(new_text):
        return new_text

    with gr.Blocks() as demo:
        gr.Markdown("## 图片检索系统")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    search_type = gr.Radio(
                        choices=["文字搜索", "图片搜索"],
                        value="文字搜索",
                        label="搜索类型"
                    )
                text_search_col = gr.Column(visible=True)
                image_search_col = gr.Column(visible=False)

                # 文字搜索组件
                with text_search_col:
                    with gr.Row():
                        text_input = gr.Textbox(label="输入搜索文本")
                    with gr.Row():
                        search_btn = gr.Button("文字搜索")

                    # 标签区域
                    with gr.Row():
                        gr.Markdown("### 标签选择")
                    tags = [["apple", "melon", "Tomato"],
                            ["potato", "Milk", "yogurt"]]
                    for row in tags:
                        with gr.Row():
                            for tag in row:
                                gr.Button(tag).click(
                                    fn=lambda t=tag: update_text(t),
                                    outputs=text_input
                                )

                # 图片搜索组件
                with image_search_col:
                    with gr.Row():
                        image_input = gr.Image(
                            label="上传或拖拽图片",
                            type="filepath",
                            height=200,
                            elem_id="main_image_input"
                        )
                    with gr.Row():
                        image_search_btn = gr.Button("图片搜索")

                    # 随机图片展示区域
                    with gr.Row():
                        gr.Markdown(
                            "### 随机示例图片（点击搜索）")
                    with gr.Row():
                        sample_images = [
                            gr.Image(
                                type="filepath",
                                height=150,  # 增大高度
                                interactive=False,
                                show_download_button=False,
                                show_share_button=False,
                                container=False,
                                show_label=False,
                            ) for _ in range(3)
                        ]

                    # 刷新按钮
                    with gr.Row():
                        refresh_btn = gr.Button("刷新示例图片")

                        # 事件绑定调整
                        for img_component in sample_images:
                            img_component.select(
                                fn=lambda x: x,
                                inputs=img_component,
                                outputs=image_input
                            )

                        # 刷新按钮事件
                        refresh_btn.click(
                            fn=refresh_samples,
                            outputs=sample_images
                        )

            # 结果展示区域
            with gr.Column():
                with gr.Row():
                    top_k_input = gr.Slider(
                        label="返回图片数量",
                        minimum=1,
                        maximum=15,
                        value=Config.top_k,
                        step=1,
                        interactive=True
                    )
                with gr.Row():
                    gallery = gr.Gallery(
                        label="搜索结果",
                        columns=5,
                        preview=True
                    )

        # 切换搜索类型时的显示逻辑
        def toggle_search_type(search_type):
            text_visible = search_type == "文字搜索"
            image_visible = not text_visible
            return [
                {"visible": text_visible, "__type__": "update"},
                {"visible": image_visible, "__type__": "update"}
            ]

        search_type.change(
            fn=toggle_search_type,
            inputs=search_type,
            outputs=[text_search_col, image_search_col]
        )

        # 初始化时加载图片
        demo.load(
            fn=refresh_samples,
            outputs=sample_images
        )
        # 切换搜索类型时刷新
        search_type.change(
            fn=lambda: refresh_samples(),
            outputs=sample_images
        )

        # 绑定不同搜索事件
        search_btn.click(
            fn=search_text,
            inputs=[text_input, top_k_input],
            outputs=gallery
        )
        image_search_btn.click(
            fn=search_image,
            inputs=[image_input, top_k_input],
            outputs=gallery
        )

    return demo

if __name__ == "__main__":
    # 处理所有图片
    process_images()
    demo = create_interface()
    demo.launch(server_name="0.0.0.0" if os.name != 'nt' else "127.0.0.1")