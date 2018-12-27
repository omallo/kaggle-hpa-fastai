import os
import xml.etree.ElementTree as ET
from multiprocessing.pool import Pool

import cv2
import pandas as pd
import requests

PROTEINS = [
    'nucleoplasm',
    'nuclear membrane',
    'nucleoli',
    'nucleoli fibrillar center',
    'nuclear speckles',
    'nuclear bodies',
    'endoplasmic reticulum',
    'golgi apparatus',
    'peroxisomes',
    'endosomes',
    'lysosomes',
    'intermediate filaments',
    'actin filaments',
    'focal adhesion sites',
    'microtubules',
    'microtubule ends',
    'cytokinetic bridge',
    'mitotic spindle',
    'microtubule organizing center',
    'centrosome',
    'lipid droplets',
    'plasma membrane',
    'cell junctions',
    'mitochondria',
    'aggresome',
    'cytosol',
    'cytoplasmic bodies',
    'rods & rings',
    # ---
    'midbody',
    'cleavage furrow',
    'nucleus',
    'vesicles',
    'midbody ring'
]

IMAGE_URL_PREFIX = 'http://v18.proteinatlas.org/images/'
IMAGE_URL_SUFFIX = '_blue_red_green.jpg'

BASE_HPA_DIR = '/storage/kaggle/hpa'
BASE_HPA_EXT_DIR = '/storage/kaggle/hpa_external'


def download_xml(gene_id):
    dst_file_path = '{}/xmls/{}.xml'.format(BASE_HPA_EXT_DIR, gene_id)
    if not os.path.isfile(dst_file_path):
        url = 'https://v18.proteinatlas.org/{}.xml'.format(gene_id)
        r = requests.get(url, allow_redirects=True)
        open(dst_file_path, 'wb').write(r.content)


def download_image(image_id):
    v18_url = 'http://v18.proteinatlas.org/images/'
    img = image_id.split('_')
    for color in ['red', 'green', 'blue', 'yellow']:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = image_id + "_" + color + ".jpg"
        dst_file_path = '{}/images/{}'.format(BASE_HPA_EXT_DIR, img_name)
        if not os.path.isfile(dst_file_path):
            img_url = v18_url + img_path
            r = requests.get(img_url, allow_redirects=True)
            open(dst_file_path, 'wb').write(r.content)
            convert_to_png(dst_file_path)


def convert_to_png(src_file):
    basename = os.path.basename(src_file)
    dst_file = '{}/pngs/{}.png'.format(BASE_HPA_EXT_DIR, basename[:-4])
    if not os.path.isfile(dst_file):
        channel_name = src_file.split('_')[-1][:-4]
        image = load_image_channel(src_file, 512, channel_name)
        cv2.imwrite(dst_file, image)


def load_image_channel(file_path, image_size, channel_name):
    channel = cv2.imread(file_path)
    if channel is None:
        error_message = 'could not load image: "{}"'.format(file_path)
        print(error_message, flush=True)
        raise Exception(error_message)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if channel_name == 'red':
        channel = channel[:, :, 2]
    elif channel_name == 'green':
        channel = channel[:, :, 1]
    elif channel_name == 'blue':
        channel = channel[:, :, 0]
    elif channel_name == 'yellow':
        channel = (0.5 * channel[:, :, 2] + 0.5 * channel[:, :, 1]).astype(np.uint8)
    else:
        raise Exception('unexpected channel name "{}"'.format(channel_name))

    return channel


def parse_xml(gene_id):
    result = []

    tree = ET.parse('{}/xmls/{}.xml'.format(BASE_HPA_EXT_DIR, gene_id))
    sub_assay_elements = tree.findall(".//cellExpression/subAssay[@type='human']")
    for sub_assay_element in sub_assay_elements:
        verification = sub_assay_element.findall('./verification')[0].text.lower()
        if verification == 'approved':
            data_elements = sub_assay_element.findall("./data")
            for data_element in data_elements:
                location_elements = data_element.findall('./location')
                # locations = [(l.attrib['GOId'], l.text) for l in location_elements]
                locations = []
                for location_element in location_elements:
                    location = PROTEINS.index(location_element.text)
                    if location < 28:
                        locations.append(location)
                if len(locations) > 0:
                    image_url_elements = data_element.findall("./assayImage/image/imageUrl")
                    for image_url_element in image_url_elements:
                        image_url = image_url_element.text
                        if not image_url.startswith(IMAGE_URL_PREFIX) or not image_url.endswith(IMAGE_URL_SUFFIX):
                            raise Exception('unexpected image URL "{}"'.format(image_url))
                        image_id = image_url[len(IMAGE_URL_PREFIX):-len(IMAGE_URL_SUFFIX)].replace('/', '_')
                        if not os.path.isfile('{}/images/{}_green.jpg'.format(BASE_HPA_EXT_DIR, image_id)):
                            # raise Exception('unexpected image ID "{}" from URL "{}"'.format(image_id, image_url))
                            #  print('images for ID "{}" were not downloaded'.format(image_id), flush=True)
                            continue
                        result.append((image_id, locations))

    return result


if __name__ == "__main__":
    os.makedirs('{}/images'.format(BASE_HPA_EXT_DIR), exist_ok=True)
    os.makedirs('{}/pngs'.format(BASE_HPA_EXT_DIR), exist_ok=True)
    os.makedirs('{}/xmls'.format(BASE_HPA_EXT_DIR), exist_ok=True)

    df = pd.read_csv('./analysis/subcellular_location.tsv', sep='\t', index_col='Gene')

    with Pool(64) as pool:
        pool.map(download_xml, df.index.tolist())

    sample_ids = []
    sample_targets = []
    with Pool(64) as pool:
        for samples in pool.map(parse_xml, df.index.tolist()):
            for sample in samples:
                sample_ids.append(sample[0])
                sample_targets.append(sample[1])
    print(len(sample_ids))

    with Pool(64) as pool:
        pool.map(download_image, sample_ids)

    train_df = pd.read_csv('{}/train.csv'.format(BASE_HPA_DIR), index_col='Id')
    external_df = pd.DataFrame(index=sample_ids, data={'Target': [' '.join(list(map(str, t))) for t in sample_targets]})
    combined_df = pd.concat([train_df, external_df])
    combined_df.to_csv('{}/train_extended.csv'.format(BASE_HPA_DIR), index_label='Id')
