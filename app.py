import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageFilter, ImageEnhance
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def crop_image(image, left, upper, right, lower):
    return image.crop((left, upper, right, lower))


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


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

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 打开图片获取尺寸
        img = Image.open(file_path)
        width, height = img.size

        return render_template('edit.html',
                               original_image=filename,
                               width=width,
                               height=height)

    return redirect(request.url)


@app.route('/process', methods=['POST'])
def process_image():
    original_image = request.form['original_image']
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)

    img = Image.open(original_path)

    # 处理裁剪
    if 'crop' in request.form:
        left = int(request.form['left'])
        upper = int(request.form['upper'])
        right = int(request.form['right'])
        lower = int(request.form['lower'])
        img = crop_image(img, left, upper, right, lower)

    # 处理亮度调整
    if 'brightness' in request.form:
        brightness = float(request.form['brightness'])
        img = adjust_brightness(img, brightness)

    # 处理对比度调整
    if 'contrast' in request.form:
        contrast = float(request.form['contrast'])
        img = adjust_contrast(img, contrast)

    # 处理滤镜
    if 'filter' in request.form:
        filter_type = request.form['filter']
        if filter_type == 'blur':
            img = apply_blur(img)
        elif filter_type == 'sharpen':
            img = apply_sharpen(img)

    # 保存处理后的图片
    processed_filename = 'processed_' + original_image
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    img.save(processed_path)

    return render_template('result.html',
                           original_image=original_image,
                           processed_image=processed_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)