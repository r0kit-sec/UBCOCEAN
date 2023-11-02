import json
import subprocess

with open('updated_image_ids.json', 'r') as json_file:
    file_list = json.load(json_file)

competition_name = 'UBC-OCEAN'

def download(is_thumbnail=False):
    for file_name in file_list:
        if is_thumbnail:
            command = f'kaggle competitions download -c {competition_name} -f train_thumbnails/{file_name}_thumbnail.png -p input/train_thumbnails/ -o'
        else:
            command = f'kaggle competitions download -c {competition_name} -f train_images/{file_name}.png -p input/train_images/ -o'
        try:
            subprocess.run(command, shell=True, check=True)
            print(f'Successfully downloaded {file_name}.png')
        except subprocess.CalledProcessError as e:
            print(f'Error downloading {file_name}.png: {e}')

    print('Download process completed.')

download(is_thumbnail=True)
download(is_thumbnail=False)