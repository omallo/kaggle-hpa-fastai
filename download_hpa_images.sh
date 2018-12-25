#!/usr/bin/env bash

set -e
set -o pipefail

function install_dependencies() {
  add-apt-repository ppa:deadsnakes/ppa >/dev/null
  apt-get update >/dev/null

  apt-get -y install python3.6 python3.6-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
  rm -rf /var/lib/apt/lists/*

  pip -q install virtualenv
  virtualenv env --python=python3.6
  . env/bin/activate

  pip -q install -r requirements.txt
}

install_dependencies

python download_hpa_images.py
