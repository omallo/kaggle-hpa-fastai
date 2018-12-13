#!/usr/bin/env bash

set -e
set -o pipefail

if [ -z $1 ]
then
  echo "missing run name"
  exit 1
fi

trap archive_artifacts EXIT

function install_dependencies() {
  apt-get update >/dev/null

  add-apt-repository ppa:deadsnakes/ppa >/dev/null
  apt-get -y update >/dev/null
  apt-get -y install python3.6 python3-pip >/dev/null

  apt-get -y install libsm-dev libxrender1 libxext6 zip git >/dev/null
  rm -rf /var/lib/apt/lists/*

  pip3 -q install virtualenv
  virtualenv env --python=python3.6
  . env/bin/activate

  pip3 -q install -r requirements.txt
}

function archive_artifacts() {
  if [ -d /artifacts/logs ]
  then
    ( cd /artifacts && zip -q -r logs.zip logs )
    rm -rf /artifacts/logs
  fi

  rm -rf /storage/models/hpa/${RUN_NAME}
  mkdir -p /storage/models/hpa/${RUN_NAME}
  cp -r /artifacts/* /storage/models/hpa/${RUN_NAME}
}

RUN_NAME=$1

while (( "$#" ))
do
  case "$1" in
  --)
    shift
    break
    ;;
  *)
    shift
    ;;
  esac
done

install_dependencies

printf "commit: $(git rev-parse HEAD)\n\n" | tee -a /artifacts/out.log

python -m cProfile -o /artifacts/train.prof train.py $* 2>/artifacts/err.log | tee -a /artifacts/out.log
