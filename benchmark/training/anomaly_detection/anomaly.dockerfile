FROM python:3.6.5

WORKDIR /nfs/home/abenalla/caryle/caryle/tiny/benchmark/training/anomaly_detection

COPY . /nfs/home/abenalla/caryle/caryle/tiny/benchmark/training/anomaly_detection

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
