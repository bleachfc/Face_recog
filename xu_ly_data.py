from sklearn.model_selection import train_test_split
import os
import shutil

# Hàm để chia dữ liệu thành các thư mục train và test
def split_data(input_dir, train_dir, test_dir, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        images = os.listdir(class_dir)
        
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        class_train_dir = os.path.join(train_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        
        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)
        if not os.path.exists(class_test_dir):
            os.makedirs(class_test_dir)
        
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(class_train_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(class_test_dir, img))

# Đường dẫn tới thư mục chứa ảnh gốc
input_dir = 'Face_data/raw'
train_dir = 'Face_data/train'
test_dir = 'Face_data/test'

# Chia dữ liệu
split_data(input_dir, train_dir, test_dir, test_size=0.2)
