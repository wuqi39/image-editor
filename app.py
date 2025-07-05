import os
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageFilter, ImageEnhance
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'models'  # 新增：本地模型文件夹
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 创建必要文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)  # 确保模型文件夹存在

# 全局存储预加载的超分辨率模型
preloaded_upscalers = {}


# 初始化Real-ESRGAN超分辨率模型（优先使用本地模型）
def init_upscaler(scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 本地模型路径（手动下载后存放的位置）
    local_model_paths = {
        2: os.path.join(app.config['MODEL_FOLDER'], 'RealESRGAN_x2plus.pth'),
        4: os.path.join(app.config['MODEL_FOLDER'], 'RealESRGAN_x4plus.pth'),
        8: os.path.join(app.config['MODEL_FOLDER'], 'RealESRGAN_x8plus.pth')  # 本地8倍模型
    }
    # 官方模型链接（备用）
    official_model_paths = {
        2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        8: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x8plus.pth'  # 最新尝试链接
    }

    # 优先使用本地模型，若不存在则使用官方链接
    if os.path.exists(local_model_paths[scale]):
        model_path = local_model_paths[scale]
        print(f"使用本地模型：{model_path}")
    else:
        model_path = official_model_paths[scale]
        print(f"本地模型不存在，使用官方链接：{model_path}")

    # 定义模型结构
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False if device.type == 'cpu' else True,
        device=device
    )
    return upsampler, device


# 预加载模型（忽略8倍模型失败，不影响其他功能）
def preload_models():
    supported_scales = [2, 4, 8]
    for scale in supported_scales:
        try:
            print(f"预加载 {scale}倍放大模型...")
            upsampler, device = init_upscaler(scale)
            preloaded_upscalers[scale] = (upsampler, device)
            print(f"{scale}倍模型加载成功（设备：{device}）")
        except Exception as e:
            print(f"预加载 {scale}倍模型失败：{str(e)}")
            # 仅8倍模型允许失败，2/4倍模型失败则提示关键错误
            if scale not in [8]:
                print(f"错误：{scale}倍模型为核心功能，加载失败将导致该功能不可用")


preload_models()


# 超分辨率放大函数（增加8倍模型失败的降级处理）
def upscale_image(image, scale):
    # 若8倍模型加载失败，自动降级为4倍
    if scale == 8 and 8 not in preloaded_upscalers:
        print("警告：8倍模型加载失败，自动降级为4倍放大")
        scale = 4

    if scale not in preloaded_upscalers:
        raise ValueError(f"不支持的放大倍数：{scale}，可用倍数：{list(preloaded_upscalers.keys())}")

    upsampler, device = preloaded_upscalers[scale]

    # 处理透明通道（转换为RGB）
    if image.mode in ('RGBA', 'LA'):
        background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
        background.paste(image, image.split()[-1])
        image = background.convert('RGB')
    else:
        image = image.convert('RGB')

    img_np = np.array(image)[:, :, ::-1]  # RGB转BGR
    output, _ = upsampler.enhance(img_np, outscale=scale)
    output_img = Image.fromarray(output[:, :, ::-1])  # BGR转RGB
    return output_img


# 以下为原有函数（保持不变）
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def crop_image(image, left, upper, right, lower):
    return image.crop((left, upper, right, lower))


def adjust_brightness(image, factor):
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image, factor):
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_blur(image):
    return image.filter(ImageFilter.BLUR)


def apply_sharpen(image):
    return image.filter(ImageFilter.SHARPEN)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '' or not (file and allowed_file(file.filename)):
        return redirect(request.url)

    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    with Image.open(file_path) as img:
        width, height = img.size

    return render_template('edit.html', original_image=filename, width=width, height=height)


@app.route('/process', methods=['POST'])
def process_image():
    original_image = request.form['original_image']
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)

    with Image.open(original_path) as img:
        left = int(request.form['left'])
        upper = int(request.form['upper'])
        right = int(request.form['right'])
        lower = int(request.form['lower'])
        img = crop_image(img, left, upper, right, lower)

        brightness = float(request.form['brightness'])
        img = adjust_brightness(img, brightness)

        contrast = float(request.form['contrast'])
        img = adjust_contrast(img, contrast)

        filter_type = request.form.get('filter')
        if filter_type == 'blur':
            img = apply_blur(img)
        elif filter_type == 'sharpen':
            img = apply_sharpen(img)

        upscale_factor = int(request.form.get('upscale_factor', 2))
        if upscale_factor > 1:
            try:
                img = upscale_image(img, upscale_factor)
            except Exception as e:
                return f"图片放大失败：{str(e)}", 500

    processed_filename = 'processed_' + original_image
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    img.save(processed_path)

    return render_template('result.html', original_image=original_image, processed_image=processed_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)