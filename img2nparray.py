import os
import numpy as np
from PIL import Image

def images_to_nparray_with_labels(root_folder, image_size=(224, 224)):
    x_train = []
    y_train = []
    label_map = {}  # Klasör isimlerini sayısal etiketlere eşlemek için

    for idx, subfolder in enumerate(os.listdir(root_folder)):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_map[subfolder] = idx  # Klasör ismi için sayısal etiket ata
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                    image_path = os.path.join(subfolder_path, filename)
                    try:
                        with Image.open(image_path) as img:
                            img = img.resize(image_size)
                            image_array = np.array(img)
                            x_train.append(image_array)
                            y_train.append(idx)  # Sayısal etiket ekle
                    except Exception as e:
                        print(f"Hata oluştu: {image_path} - {e}")

    return np.array(x_train), np.array(y_train)

# Örnek kullanım
if __name__ == "__main__":
    root_folder = "./Data" 
    x_train, y_train = images_to_nparray_with_labels(root_folder)  
    np.savez("Data_224x224_garbage_6Class.npz", x_train=x_train, y_train=y_train)    
    print(f"Toplam {len(x_train)} görüntü dönüştürüldü ve Data_224x224_garbage_6Class.npz dosyasına kaydedildi.")
