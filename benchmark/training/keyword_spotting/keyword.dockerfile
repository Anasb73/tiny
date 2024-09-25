FROM tensorflow/tensorflow:2.9.3-gpu

WORKDIR /nfs/home/abenalla/caryle/caryle/tiny/benchmark/training/keyword_spotting

COPY . /nfs/home/abenalla/caryle/caryle/tiny/benchmark/training/keyword_spotting

RUN pip install --upgrade pip

RUN pip install matplotlib tensorflow_datasets pydub
