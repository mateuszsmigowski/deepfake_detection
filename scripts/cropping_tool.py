import os
from pathlib import Path
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from PIL import Image

def crop_face_with_margin(image_path: Path, output_path: Path, mtcnn: MTCNN, target_size=(256, 256), margin_ratio=0.3):
    try:
        img = Image.open(image_path).convert('RGB')
        
        boxes, _ = mtcnn.detect(img)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = box
            
            w = x2 - x1
            h = y2 - y1
            
            margin_x = w * margin_ratio
            margin_y = h * margin_ratio
            
            img_w, img_h = img.size
            new_x1 = max(0, int(x1 - margin_x))
            new_y1 = max(0, int(y1 - margin_y))
            new_x2 = min(img_w, int(x2 + margin_x))
            new_y2 = min(img_h, int(y2 + margin_y))
            
            face_img = img.crop((new_x1, new_y1, new_x2, new_y2))
            
            face_img = face_img.resize(target_size, Image.Resampling.LANCZOS)
            
            face_img.save(output_path)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Błąd przetwarzania {image_path.name}: {str(e)}")
        return False

def preprocess_dataset(src_dir: str, dst_dir: str):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=False, device=device)
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    all_files = [p for p in src_path.rglob('*') if p.suffix.lower() in image_extensions]
    
    print(f"Znaleziono {len(all_files)} obrazów do przetworzenia.")
    
    skipped_count = 0
    success_count = 0
    
    for img_file in tqdm(all_files, desc="Wycinanie twarzy"):
        relative_path = img_file.relative_to(src_path)
        output_file = dst_path / relative_path
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        success = crop_face_with_margin(img_file, output_file, mtcnn, target_size=(256, 256))
        
        if success:
            success_count += 1
        else:
            skipped_count += 1
            
    print(f"Przetwarzanie zakończone!")
    print(f"Sukces: {success_count} obrazów.")
    print(f"Pominięto (brak twarzy): {skipped_count} obrazów.")

if __name__ == "__main__":
    SRC = "/Volumes/DyskSSD1T/Datasets/FaceForensics++ Extracted Dataset (C23)/FF++C32-Frames"
    DST = "/Volumes/DyskSSD1T/Datasets/FaceForensics++ Extracted Dataset (C23)/FF++C32-Frames_cropped"
    preprocess_dataset(SRC, DST)