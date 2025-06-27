FROM python:3 AS conda-base

WORKDIR /tmp
RUN  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /root/anaconda3 -b -u

RUN /root/anaconda3/bin/conda init --all

# ----------------------------------------------------------------------------------------

FROM conda-base AS spinach

## copy custom project
COPY . /app

WORKDIR /app

## install anaconda and pip deps
RUN /root/anaconda3/bin/conda env create -f conda_env.yaml

COPY deploy/app/app.py /app

EXPOSE 8085
ENTRYPOINT [ "bash" ]
CMD [ "-lc", "conda activate text2sparql &&  redis-server --port 6379 --daemonize yes && gunicorn --workers 4 --bind 0.0.0.0:8085 --timeout 180 'app:create_app()'"]