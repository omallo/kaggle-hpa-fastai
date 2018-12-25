import glob
import os
from multiprocessing.pool import Pool

import pandas as pd
import requests


def download(pid, sp, ep):
    colors = ['red', 'green', 'blue', 'yellow']
    DIR = "/storage/kaggle/hpa_external/images/"
    v18_url = 'http://v18.proteinatlas.org/images/'
    imgList = pd.read_csv("/storage/kaggle/hpa_external/HPAv18RBGY_wodpl.csv")
    for i in imgList['Id'][sp:ep]:
        img = i.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = i + "_" + color + ".jpg"
            if not os.path.isfile(DIR + img_name):
                print('fetching image "{}"'.format(img_name), flush=True)
                img_url = v18_url + img_path
                r = requests.get(img_url, allow_redirects=True)
                open(DIR + img_name, 'wb').write(r.content)


def run_proc(name, sp, ep):
    print('Run child process %s (%s) sp:%d ep: %d' % (name, os.getpid(), sp, ep), flush=True)
    download(name, sp, ep)
    print('Run child process %s done' % (name), flush=True)


def do_download():
    print('Parent process %s.' % os.getpid(), flush=True)
    img_list = pd.read_csv("/storage/kaggle/hpa_external/HPAv18RBGY_wodpl.csv")['Id']
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
    for f in glob.glob('/storage/kaggle/hpa_external/images/*.jpg'):
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


if __name__ == "__main__":
    do_download()
    do_analyze()
