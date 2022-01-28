FROM ubuntu:20.04

RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends tzdata

#RUN apt update

RUN apt-get install -y build-essential && apt install -y build-essential\
 cmake\ 
 libboost-system-dev\ 
 libboost-thread-dev\ 
 libboost-program-options-dev\ 
 libboost-test-dev\ 
 libeigen3-dev\ 
 zlib1g-dev\ 
 libbz2-dev\ 
 liblzma-dev\
 && apt-get install -y git-lfs wget

RUN apt-get install -y python3-pip && ln -s /usr/bin/python3 /usr/bin/python 
#&& ln -s /usr/bin/python3.8 /usr/bin/python

RUN pip3 install datasets\
 transformers\
 typer\
 pyctcdecode\
 torch\
 https://github.com/kpu/kenlm/archive/master.zip

WORKDIR ngram

RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
RUN mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2

COPY . .

RUN GIT_LFS_SKIP_SMUDGE=1

ENTRYPOINT ["sh", "process.sh"]
#CMD ["bash"]