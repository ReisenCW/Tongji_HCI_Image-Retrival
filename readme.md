## 如何运行此项目

### 环境要求  
- Python 3.12+  
- CUDA 11.x（如有可用 GPU）  
- Upstash 向量数据库账号（[注册地址](https://upstash.com/)）  
- 数据集文件（[下载地址](https://github.com/marcusklasson/GroceryStoreDataset)）  

---

### 运行步骤  
#### 1. 安装依赖  
```bash
# 安装核心依赖库
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install upstash-vector gradio loguru transformers numpy Pillow
```

#### 2. 准备数据集  
1. 下载数据集到 `./GroceryStoreDataset`

#### 3. 配置 Upstash 服务（可选）
1. 在 [Upstash 控制台](https://console.upstash.com/) 创建向量数据库实例  
2. 修改 `Config` 类中的服务地址和令牌：  
   ```python
   class Config:
       upstash_url = "YOUR_UPSTASH_ENDPOINT_URL"  # 替换为你的链接
       token = "YOUR_UPSTASH_TOKEN"               # 替换为你的令牌
   ```
3. 注意事项：大陆网络环境需要配置网络代理才能连接到 Upstash 服务

#### 4. 运行项目  
```bash
python main.py
```

---

### 功能说明  
- **首次运行**：  
  自动执行 `process_images()`，将数据集图像编码并上传至 Upstash 向量数据库（约需 10-30 秒，依赖 GPU 性能）。  
- **后续运行**：  
  直接启动 Gradio 交互界面，默认地址为 `http://localhost:7860`。

---

通过以上步骤，即可快速部署并体验完整的图像检索系统。  