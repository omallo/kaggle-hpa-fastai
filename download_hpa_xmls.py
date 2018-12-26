import os
import xml.etree.ElementTree as ET
from multiprocessing.pool import Pool

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


def download_xml(gene_id):
    url = 'https://v18.proteinatlas.org/{}.xml'.format(gene_id)
    r = requests.get(url, allow_redirects=True)
    open('../../hpa_external/xmls/{}.xml'.format(gene_id), 'wb').write(r.content)


def parse_xml(gene_id):
    result = []

    tree = ET.parse('../../hpa_external/xmls/{}.xml'.format(gene_id))
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
                        if not os.path.isfile('../../hpa_external/images/{}_green.jpg'.format(image_id)):
                            # raise Exception('unexpected image ID "{}" from URL "{}"'.format(image_id, image_url))
                            #  print('images for ID "{}" were not downloaded'.format(image_id), flush=True)
                            continue
                        result.append((image_id, locations))

    return result


if __name__ == "__main__":
    df = pd.read_csv('./analysis/subcellular_location.tsv', sep='\t', index_col='Gene')

    if False:
        with Pool(16) as pool:
            pool.map(download_xml, df.index.tolist())

    sample_ids = []
    sample_targets = []
    with Pool(16) as pool:
        for samples in pool.map(parse_xml, df.index.tolist()):
            for sample in samples:
                sample_ids.append(sample[0])
                sample_targets.append(sample[1])
    print(len(sample_ids))

    approved_df = pd.DataFrame(index=sample_ids, data={'Target': [' '.join(list(map(str, t))) for t in sample_targets]})
    approved_df.to_csv('../../hpa/train_extended_approved.csv', index_label='Id')
