import os
from PIL import Image


def rotate_and_save_image(image_path):

    with Image.open(image_path) as img:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(1, 4):
            rotated_img = img.rotate(90 * i, expand=True)
            new_image_name = f"{base_name}_rotate{i}.jpg"
            new_image_path = os.path.join('ece471_data/dataset/screws/train/good/class1', new_image_name)  
            rotated_img.save(new_image_path)
            print(f"Saved rotated image: {new_image_name}")

def main():

    cwd = os.getcwd()
    cwd = os.path.join(cwd, 'ece471_data/dataset/screws/train/good/class1')

    for file_name in os.listdir(cwd):
        if file_name.lower().endswith('.jpg'):
            image_path = os.path.join(cwd, file_name)
            rotate_and_save_image(image_path)

if __name__ == "__main__":
    main()