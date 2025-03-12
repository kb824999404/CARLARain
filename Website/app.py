from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
import threading
from datetime import datetime
import random
from PIL import Image, ImageDraw

import sys



app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.config['HRIG_FOLDER'] = 'tasks_hrig'  # 雨景图像生成任务文件夹
app.config['CRIG_FOLDER'] = 'tasks_crig'  # 雨纹图像生成任务文件夹
os.makedirs(app.config['HRIG_FOLDER'], exist_ok=True)
os.makedirs(app.config['CRIG_FOLDER'], exist_ok=True)

# 初始化模型
# model = HRIGNet()

# 任务状态
TASK_STATUS = {
    "PENDING": "pending",
    "COMPLETED": "completed",
    "FAILED": "failed"
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/imgs/<path:filename>', methods=['GET'])
def imgs(filename):
    return send_from_directory('static/imgs', filename)

def generate_rainy_image(task_id, background_path, rain_path, steps, use_lighten, use_blend):
    """在后台线程中生成雨景图像"""
    sys.path.append("../HRIGNet")
    from hrig_predict_module import predict_from_bg_mask
    try:
        output_path = os.path.join(app.config['HRIG_FOLDER'], task_id, 'output.png')
        print("Generating Rainy Image...")
        print(f"Steps:{steps}\tUse Lighten:{use_lighten}\tUse Blend:{use_blend}")
        predict_from_bg_mask(background_path,rain_path,output_path, steps, use_lighten, use_blend)
        update_task_status(task_id, TASK_STATUS["COMPLETED"], is_rain_pattern=False)
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        update_task_status(task_id, TASK_STATUS["FAILED"], is_rain_pattern=False)

def update_task_status(task_id, status, is_rain_pattern=False):
    """更新任务状态"""
    if is_rain_pattern:
        task_folder = os.path.join(app.config['CRIG_FOLDER'], task_id)
    else:
        task_folder = os.path.join(app.config['HRIG_FOLDER'], task_id)
    task_info_path = os.path.join(task_folder, 'task_info.json')
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)
    task_info['status'] = status
    with open(task_info_path, 'w') as f:
        json.dump(task_info, f)

@app.route('/upload', methods=['POST'])
def upload_files():
    """上传图像并创建任务"""
    if 'background' not in request.files or 'rain' not in request.files:
        return jsonify({"error": "Missing files"}), 400
    
    background_file = request.files['background']
    rain_file = request.files['rain']
    use_lighten = request.form['use_lighten']
    use_blend = request.form['use_blend']
    steps = int(request.form['steps'])
    if use_lighten == 'false':
        use_lighten = False
    else:
        use_lighten = True
    if use_blend == 'false':
        use_blend = False
    else:
        use_blend = True
    
    if background_file.filename == '' or rain_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # 创建任务文件夹
    task_id = str(uuid.uuid4())
    task_folder = os.path.join(app.config['HRIG_FOLDER'], task_id)
    os.makedirs(task_folder, exist_ok=True)
    
    # 保存上传的图像
    background_path = os.path.join(task_folder, 'background.png')
    rain_path = os.path.join(task_folder, 'rain.png')
    background_file.save(background_path)
    rain_file.save(rain_path)
    
    # 创建任务信息
    task_info = {
        "task_id": task_id,
        "task_name": f"Task {task_id[:8]}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": TASK_STATUS["PENDING"]
    }
    with open(os.path.join(task_folder, 'task_info.json'), 'w') as f:
        json.dump(task_info, f)
    
    # 启动后台任务
    threading.Thread(target=generate_rainy_image, args=(task_id, background_path, rain_path, steps, use_lighten, use_blend)).start()
    
    return jsonify(task_info), 200

@app.route('/tasks', methods=['GET'])
def get_tasks():
    """获取所有雨景图像生成任务"""
    tasks = []
    for task_id in os.listdir(app.config['HRIG_FOLDER']):
        task_folder = os.path.join(app.config['HRIG_FOLDER'], task_id)
        task_info_path = os.path.join(task_folder, 'task_info.json')
        if os.path.exists(task_info_path):
            with open(task_info_path, 'r') as f:
                task_info = json.load(f)
                tasks.append(task_info)
    return jsonify(tasks), 200

@app.route('/task/<task_id>/image/<image_type>', methods=['GET'])
def get_task_image(task_id, image_type):
    """获取任务图像（上传的图像或生成的图像）"""
    task_folder = os.path.join(app.config['HRIG_FOLDER'], task_id)
    if image_type == 'background':
        image_path = os.path.join(task_folder, 'background.png')
    elif image_type == 'rain':
        image_path = os.path.join(task_folder, 'rain.png')
    elif image_type == 'output':
        image_path = os.path.join(task_folder, 'output.png')
    else:
        return jsonify({"error": "Invalid image type"}), 400
    
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(image_path, mimetype='image/png')

def generate_rain_image(task_id,intensity,direction,randomSeed):
    """在后台线程中生成雨纹图像"""
    sys.path.append("../CRIGNet")
    from crig_predict_module import predict_rain_from_zero
    try:
        output_path = os.path.join(app.config['CRIG_FOLDER'], task_id, 'output.png')
        print("Generating Rain Image...")
        print(f"Intensity:{intensity}\tDirection:{direction}\tSeed:{randomSeed}")
        predict_rain_from_zero(output_path,intensity,direction,randomSeed)
        update_task_status(task_id, TASK_STATUS["COMPLETED"], is_rain_pattern=True)
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        update_task_status(task_id, TASK_STATUS["FAILED"], is_rain_pattern=True)

# 生成雨纹图像
@app.route('/generate_rain_pattern', methods=['POST'])
def generate_rain_pattern():
    """生成雨纹图像"""
    # 获取请求参数
    intensity = float(request.form.get('rain_intensity'))
    direction = float(request.form.get('rain_direction'))
    randomSeed = int(request.form.get('randomSeed'))
    

    # 创建任务文件夹
    task_id = str(uuid.uuid4())
    task_folder = os.path.join(app.config['CRIG_FOLDER'], task_id)
    os.makedirs(task_folder, exist_ok=True)

    # 创建任务信息
    task_info = {
        "task_id": task_id,
        "task_name": f"Rain Pattern Task {task_id[:8]}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": TASK_STATUS["PENDING"]
    }
    with open(os.path.join(task_folder, 'task_info.json'), 'w') as f:
        json.dump(task_info, f)

    # 启动后台任务
    threading.Thread(target=generate_rain_image, args=(task_id,intensity,direction,randomSeed)).start()

    return jsonify(task_info), 200


# 新增接口：获取所有雨纹图像生成任务
@app.route('/rain_pattern_tasks', methods=['GET'])
def get_rain_pattern_tasks():
    """获取所有雨纹图像生成任务"""
    rain_pattern_tasks = []
    for task_id in os.listdir(app.config['CRIG_FOLDER']):
        task_folder = os.path.join(app.config['CRIG_FOLDER'], task_id)
        task_info_path = os.path.join(task_folder, 'task_info.json')
        if os.path.exists(task_info_path):
            with open(task_info_path, 'r') as f:
                task_info = json.load(f)
                # 筛选雨纹图像生成任务
                if "Rain Pattern" in task_info["task_name"]:
                    rain_pattern_tasks.append(task_info)
    return jsonify(rain_pattern_tasks), 200


@app.route('/rain_pattern_tasks/<task_id>/image/<image_type>', methods=['GET'])
def get_task_image_rain(task_id, image_type):
    """获取任务图像（上传的图像或生成的图像）"""
    task_folder = os.path.join(app.config['CRIG_FOLDER'], task_id)
    if image_type == 'output':
        image_path = os.path.join(task_folder, 'output.png')
    else:
        return jsonify({"error": "Invalid image type"}), 400
    
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5088)