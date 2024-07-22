from PIL import Image, ExifTags
# Add ExifTags, pip3 install piexif
import os

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def format_name(name):
    return ' '.join([word.capitalize() for word in name.split()])

def save_image(images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for image in images:
        img = Image.open(image)
        # Handle image rotation based on EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation = exif.get(orientation, 1)
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # Cases: image don't have getexif
            pass

        # img.thumbnail((800, 800))  # Mengubah ukuran gambar dengan mempertahankan rasio aspek
        img.save(os.path.join(folder_path, image.filename))

def get_folders_info(upload_folder):
    folders_info = []
    for folder_name in os.listdir(upload_folder):
        folder_path = os.path.join(upload_folder, folder_name)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            folders_info.append((folder_name, len(image_files)))
    return folders_info