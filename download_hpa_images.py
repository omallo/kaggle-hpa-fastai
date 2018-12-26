import glob
import os
from multiprocessing.pool import Pool

import imagehash
import pandas as pd
import requests
from PIL import Image


def download(pid, sp, ep):
    colors = ['red', 'green', 'blue', 'yellow']
    DIR = "../../hpa_external/images/"
    v18_url = 'http://v18.proteinatlas.org/images/'
    imgList = pd.read_csv("../../hpa_external/HPAv18RBGY_wodpl.csv")
    for i in imgList['Id'][sp:ep]:
        img = i.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = i + "_" + color + ".jpg"
            if not os.path.isfile(DIR + img_name):
                # print('fetching image "{}"'.format(img_name), flush=True)
                img_url = v18_url + img_path
                r = requests.get(img_url, allow_redirects=True)
                open(DIR + img_name, 'wb').write(r.content)


def run_proc(name, sp, ep):
    print('Run child process %s (%s) sp:%d ep: %d' % (name, os.getpid(), sp, ep), flush=True)
    download(name, sp, ep)
    print('Run child process %s done' % (name), flush=True)


def do_download():
    print('Parent process %s.' % os.getpid(), flush=True)
    img_list = pd.read_csv("../../hpa_external/HPAv18RBGY_wodpl.csv")['Id']
    list_len = len(img_list)
    process_num = 100
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(run_proc, args=(str(i), int(i * list_len / process_num), int((i + 1) * list_len / process_num)))
    print('Waiting for all subprocesses done...', flush=True)
    p.close()
    p.join()
    print('All subprocesses done.', flush=True)


def do_analyze():
    colors = ['red', 'green', 'blue', 'yellow']
    id_colors = {}
    for f in glob.glob('../../hpa_external/images/*.jpg'):
        b = os.path.basename(f)
        for c in colors:
            s = '_{}.jpg'.format(c)
            if b.endswith(s):
                id = b[:-len(s)]
                ic = id_colors.setdefault(id, [])
                ic.append(c)
                id_colors[id] = ic

    print('found {} samples'.format(len(id_colors)), flush=True)

    for k, v in id_colors.items():
        if len(v) != len(colors):
            print('sample "{}" only has colors {}'.format(k, v), flush=True)


def hash_image(file_name):
    return imagehash.phash(Image.open(file_name))


def hash_images(file_names):
    with Pool(16) as pool:
        return [h for h in pool.map(hash_image, file_names)]


def do_find_similar_images():
    print('hashing external images', flush=True)
    external_imgs = glob.glob('../../hpa_external/images/*_green.jpg')
    external_hashes = hash_images(external_imgs)
    images = {}
    for img, hash in zip(external_imgs, external_hashes):
        id = os.path.basename(img)[:-len('_green.jpg')]
        images[hash] = images.get(hash, []) + [id]

    print('finding duplicates in train set', flush=True)
    train_duplicates = []
    train_imgs = glob.glob('../../hpa/train/*_green.png')
    train_hashes = hash_images(train_imgs)
    for img, hash in zip(train_imgs, train_hashes):
        id = os.path.basename(img)[:-len('_green.png')]
        if hash in images:
            train_duplicates.append(','.join([*images[hash], id]))
    print('\n'.join(train_duplicates), flush=True)

    print('finding duplicates in test set', flush=True)
    test_duplicates = []
    test_imgs = glob.glob('../../hpa/test/*_green.png')
    test_hashes = hash_images(test_imgs)
    for img, hash in zip(test_imgs, test_hashes):
        id = os.path.basename(img)[:-len('_green.png')]
        if hash in images:
            test_duplicates.append(','.join([*images[hash], id]))
    print('\n'.join(test_duplicates), flush=True)


if __name__ == "__main__":
    do_download()
    do_analyze()
    # do_find_similar_images()
