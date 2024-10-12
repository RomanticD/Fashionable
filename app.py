import sys
import os
from pathlib import Path


# 设置 Python 路径
current_dir = Path(__file__).parent.resolve()
groundingdino_path = current_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# 导入必要的库
import streamlit as st
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image


class ImageOnlyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image = self.transform(image)
        return image


transform = T.Compose([
    ImageOnlyTransform(T.RandomResize([800], max_size=1333)),
    ImageOnlyTransform(T.ToTensor()),
    ImageOnlyTransform(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
])


# 3. 定义图像处理函数（保持不变）
def split_image_vertically(image, segment_height):
    """将图像纵向分割成多段，每段的高度为 segment_height。"""
    height, width, _ = image.shape
    segments = []
    for i in range(0, height, segment_height):
        segment = image[i:i + segment_height, :, :]
        segments.append(segment)
    return segments


def combine_segments_vertically(segments, original_height, original_width):
    """将分割的图像段重新拼接成原始图像大小。"""
    combined_image = np.vstack(segments)
    return combined_image[:original_height, :original_width, :]


def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 缩放比例
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (255, 255, 255))  # 创建白色背景
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 居中粘贴
    return new_image


# 4. 定义推理函数（略微调整以适应绝对路径）
def run_inference(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD, FRAME_WINDOW):
    bboxes = []  # 存储每个 bounding box 的图像区域
    annotated_segments = []
    for segment in segments:
        print(f"Processing segment with shape: {segment.shape}")
        # 使用包装后的 transform
        try:
            segment_transformed = transform(Image.fromarray(segment), None)
            if isinstance(segment_transformed, tuple):
                segment_transformed = segment_transformed[0]
            print(f"Segment transformed successfully.")
        except Exception as e:
            st.error(f"Transform 过程中出错: {e}")
            st.stop()

        boxes, logits, phrases = predict(
            model=model,
            device="cpu",
            image=segment_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=0.25
        )
        print(f"Predict output: boxes={boxes}, logits={logits}, phrases={phrases}")  # 添加调试输出

        print(f"Calling annotate with boxes={boxes}, logits={logits}, phrases={phrases}")

        annotated_segment, detection = annotate(segment, boxes=boxes, logits=logits, phrases=phrases)
        annotated_segments.append(annotated_segment)

        # 提取每个 bounding box 的图像区域并存储在 bboxes 列表中
        for box in detection.xyxy:
            x1, y1, x2, y2 = map(int, box)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            # 过滤掉宽度小于1/5图像宽度的bbox
            if bbox_width >= max(160, segment.shape[1] / 5) and bbox_height >= bbox_width:
                bbox_image = segment[y1:y2, x1:x2]
                bboxes.append(bbox_image)

    # 拼接所有注释过的段以形成完整图像
    annotated_image = combine_segments_vertically(annotated_segments, segments[0].shape[0] * len(segments),
                                                  segments[0].shape[1])
    FRAME_WINDOW.image(annotated_image, channels='BGR')
    return bboxes


# 5. Streamlit 应用布局
st.title('服装款式相似性检测')

# Model Backbone
CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = current_dir / 'groundingdino_swint_ogc.pth'

# 检查权重文件是否存在
if not WEIGHTS_PATH.exists():
    st.error(f"权重文件未找到，请确保将 '{WEIGHTS_PATH}' 放置在项目根目录中。")
    st.stop()

# 加载模型
model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH), device='cpu')

# Transformation
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 配置参数
BOX_THRESHOLD = st.sidebar.slider('服装检测灵敏度:', min_value=0.0, max_value=1.0, value=0.3)
TEXT_PROMPT = st.text_input('Text Prompt:', value="clothes")

# 输入输出配置
gallery_path = current_dir / 'data'
st.write(gallery_path)
Prod_ID = st.text_input('Product ID:', value="id_00008000")
output_dir = os.path.join(gallery_path, Prod_ID)
os.makedirs(output_dir, exist_ok=True)

# 文件上传
upload_img_file = st.file_uploader('选择图像', type=['jpg', 'jpeg', 'png'])
FRAME_WINDOW = st.image([])

clothes_bboxes = []

if upload_img_file is not None:
    img = Image.open(upload_img_file).convert("RGB")
    image = np.asarray(img)
    # image_transformed = transform(img, None)
    try:
        segment_transformed = transform(img, None)
        print(f"Transform output: {segment_transformed}")
    # 进行解包
        if isinstance(segment_transformed, tuple):
            segment_transformed = segment_transformed[0]
    except Exception as e:
        st.error(f"Transform 过程中出错: {e}")
        st.stop()

    # 设定分段长度为图像宽度的3倍
    segment_height = image.shape[1] * 3
    segments = split_image_vertically(image, segment_height)
    FRAME_WINDOW.image(img, channels='BGR')

# 触发:服装检测
if st.button('检测页面中的服装'):
    if 'segments' not in locals():  # 检查 segments 是否已定义
        st.error("请先上传一张图像进行处理。")
    try:
        clothes_bboxes = run_inference(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD, FRAME_WINDOW)
        # 在结果图下方显示所有返回的 clothes_bboxes 图像
        st.write("检测到的服装区域：")
        for idx, bbox in enumerate(clothes_bboxes):
            st.image(bbox, caption=f"服装 {idx + 1}")
    except Exception as e:
        st.error(f"检测过程中出错: {e}")

# 添加按钮以触发相似度检测
if st.button('检测服装相似度'):
    if not clothes_bboxes:
        st.write("请先点击‘检测页面中的服装’按钮进行检测。")
    else:
        st.write("潜在的雷同款式：")
        for idx, bbox in enumerate(clothes_bboxes):
            # 将 bbox 保存为临时图像文件
            bbox_image = Image.fromarray(bbox)
            tmp_image_path = './img_tmp/intermd.jpg'
            os.makedirs('./img_tmp', exist_ok=True)
            bbox_image.save(tmp_image_path)

            # 调用 mmfashion_retrieval.py 进行检测
            os.system(f"python3 ./mmfashion_retrieval.py --input {tmp_image_path}")

            # 显示检测结果图像
            output_image_path = './output.png'
            if os.path.exists(output_image_path):
                st.image(output_image_path, caption=f"服装 {idx + 1} 相似款", use_column_width=True)
            else:
                st.write(f"未能生成服装 {idx + 1} 的相似款图像。")
