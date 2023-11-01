FROM python:3.8-buster

WORKDIR /app

RUN apt-get update && apt-get -y install --no-install-recommends git nano apt-transport-https software-properties-common \
    wget unzip ca-certificates build-essential cmake git 
RUN apt-get -y install libtbb-dev libatlas-base-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev 
RUN apt-get -y install libv4l-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev libavresample-dev 
RUN apt-get -y install x264 libtesseract-dev libgdiplus libc6-dev libc6-dev && apt-get -y clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U fastapi uvicorn imutils python-multipart pydantic easydict jwcrypto unidecode requests
RUN pip3 install -U wave scipy pyopenssl scikit-learn
RUN pip3 install -U pymongo redis unidecode pyinstaller ffmpeg gdown 
RUN apt-get update && apt-get -y install ffmpeg
RUN apt-get -y clean
RUN pip3 install "python-jose[cryptography]" "passlib[bcrypt]"

COPY / /app

#RUN wget https://vokaturi.com/doc/OpenVokaturi-4-0.zip

RUN unzip -o OpenVokaturi-4-0.zip -d OpenVokaturi-4-0
#RUN sh -c 'unzip -q OpenVokaturi-4-0.zip -d OpenVokaturi-4-0'


EXPOSE 9988

CMD [ "python3", "program.py","9988"]

#docker build -f dockerfile -t vocal-emotion-detector .

#docker run -d --restart always -p 9988:9988 --name vocal-emotion-detector_9988 vocal-emotion-detector