import os
from pathlib import Path
from ultralytics import YOLO

# モデルのロード
model = YOLO("my_yolov8m.pt")  # 学習済みモデルのパスを指定します

# 画像フォルダの指定
image_folder = "path\\to\\img_folder"
output_folder = "path\\to\\output_folder"  # 出力フォルダを指定します


# 指定した出力フォルダが存在しない場合、フォルダを作成
os.makedirs(output_folder, exist_ok=True)

# 画像フォルダ内のPNGおよびJPG画像に対して推論を実行
for image_path in Path(image_folder).glob('*.[pjJ][pnN][gG]'):
    # 画像の推論
    results = model.predict(str(image_path),imgsz=1280, conf=0.01,device=0)

    # 各画像ごとにアノテーションファイルを作成
    for result in results:
        # バウンディングボックスの情報を取得
        bboxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # アノテーションファイルのパス
        annotation_file = os.path.join(output_folder, file_name + '.txt')

        # アノテーションファイルの作成
        with open(annotation_file, 'w') as f:
            for bbox, cls in zip(bboxes, classes):
                class_id = int(cls)
                x_center = (bbox[0] + bbox[2]) / 2 / result.orig_shape[1]
                y_center = (bbox[1] + bbox[3]) / 2 / result.orig_shape[0]
                width = (bbox[2] - bbox[0]) / result.orig_shape[1]
                height = (bbox[3] - bbox[1]) / result.orig_shape[0]
                # YOLO形式で保存 (class_id, x_center, y_center, width, height)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("アノテーションファイルの作成が完了しました！")

