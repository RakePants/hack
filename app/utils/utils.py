import cv2
from fastapi import UploadFile
import aiofiles
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, ViTForImageClassification, CvtForImageClassification
from PIL import Image
import asyncio
import torch.nn.functional as F
from collections import Counter, defaultdict

model_yolo = YOLO("best.pt")

feature_extractor_vit = AutoFeatureExtractor.from_pretrained('ViT')
model_transformer_vit = ViTForImageClassification.from_pretrained('ViT')

feature_extractor_cvt = AutoFeatureExtractor.from_pretrained('cvtv2')
model_transformer_cvt = CvtForImageClassification.from_pretrained('cvtv2')

labels = ['Бетон', 'Грунт', 'Дерево', 'Кирпич']


async def predict_vit(image: Image):
    inputs = feature_extractor_vit(images=image, return_tensors="pt")
    outputs = model_transformer_vit(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return {"class": labels[predicted_class_idx],
            "prod": F.softmax(logits, dim=-1)[0, predicted_class_idx].item()}


async def predict_cvt(image: Image):
    inputs = feature_extractor_cvt(images=image, return_tensors="pt")
    outputs = model_transformer_cvt(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return {"class": labels[predicted_class_idx],
            "prod": F.softmax(logits, dim=-1)[0, predicted_class_idx].item()}


async def main(file: UploadFile):
        async with aiofiles.open(file.filename, "wb") as f:
            content = file.file.read()
            await f.write(content)
        cap = cv2.VideoCapture(file.filename)
        start_time = 120
        end_time = 140
        # Вычисляем частоту кадров в видео
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Вычисляем начальный и конечный кадры на основе времени
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Устанавливаем указатель текущего кадра на начальный
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        n = 0
        num = 0
        result_dict = {}
        while cap.isOpened():
            n += 1
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break

            if n == 24:
                num += 1
                result = model_yolo.predict(frame, conf=0.2, verbose=False)
                im_array = result[0].plot()
                im = Image.fromarray(im_array[..., ::-1])
                im.save(f'result{num}.jpg')
                image = Image.open(f"result{num}.jpg")

                async with asyncio.TaskGroup() as tg:
                    result_vit = tg.create_task(predict_vit(image))
                    result_cvt = tg.create_task(predict_cvt(image))
                result_vit = result_vit.result()
                result_cvt = result_cvt.result()
                all_values = list(result_vit.values()) + list(result_cvt.values())

                # Подсчет количества каждого значения
                value_counts = Counter(all_values)
                # Выбор значений, которые встречаются более одного раза
                duplicate_values = {value: count for value, count in value_counts.items() if count > 1}
                results = []
                if len(list(duplicate_values.values())) == 0:
                    result_dict.update({num: {"class": result_cvt.get("class"), "prod": result_cvt.get("prod")}})
                elif list(duplicate_values.values())[0] == 2:
                    predict_class = list(duplicate_values.keys())[0]
                    if result_vit.get("class") == predict_class:
                        results.append(result_vit.get("prod"))
                    if result_cvt.get("class") == predict_class:
                        results.append(result_cvt.get("prod"))
                    result_dict.update({num: {"class": predict_class, "prod": sum(results) / 2}})
                n = 0
                print(result_dict)

            cv2.waitKey(1)

        class_totals = defaultdict(lambda: {'sum': 0, 'count': 0})
        # Обходим входной словарь и собираем суммы и количество для каждого класса
        for key, value in result_dict.items():
            class_name = value['class']
            prod_value = value['prod']
            class_totals[class_name]['sum'] += prod_value
            class_totals[class_name]['count'] += 1
        # Создаем итоговый словарь с наибольшим количеством класса и средним арифметическим prod
        most_common_class = max(class_totals, key=lambda x: class_totals[x]['count'])
        # Создаем итоговый словарь только для класса с максимальным количеством
        result_dict = {'class': most_common_class,
                       'prod': class_totals[most_common_class]['sum'] / class_totals[most_common_class][
                           'count']}

        cap.release()
        return result_dict
