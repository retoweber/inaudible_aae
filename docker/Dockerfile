FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 
RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN pip3 install --user numpy scipy tensorflow-gpu==1.8.0 pandas==0.24.2 python_speech_features
RUN pip3 install librosa
RUN apt install -y libsndfile-dev
RUN apt-get install -y git
WORKDIR /home/inaudible_aae/
RUN git clone https://github.com/mozilla/DeepSpeech.git
WORKDIR /home/inaudible_aae/DeepSpeech/
RUN git checkout tags/v0.1.1
WORKDIR /home/inaudible_aae/
RUN apt install -y wget
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
RUN tar -xzf deepspeech-0.1.0-models.tar.gz
RUN pip3 install deepspeech-gpu
RUN apt install -y libav-tools libavcodec-extra
RUN pip3 install --user pydub pyxdg
WORKDIR /home/inaudible_aae/
#RUN python3 make_checkpoint.py
