# https://vokaturi.com/using-vokaturi/example-code-for-python-batch

import io
import wave
import numpy
from pydantic import ValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
import uvicorn
from jwcrypto import jwt, jwk
import struct
import platform

import os
import datetime
import math
import sys
import threading
import re

import sys
import scipy.io.wavfile


# insert at 1, 0 is the script path (or '' in REPL)
____workingDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, ____workingDir)

print("____workingDir", ____workingDir)


# OpenVokaWavMean.py
# public-domain sample code by Vokaturi, 2022-08-25
# (note that the Vokaturi library that is loaded is not public-domain)
#
# A sample script that uses the VokaturiPlus library to extract
# the emotions from a wav file on disk.
# The file can contain a mono or stereo recording.
#
# Call syntax:
#   python3 examples/OpenVokaWavMean.py path_to_sound_file.wav
#
# For the sound file hello.wav that comes with OpenVokaturi,
# the result should be:
#    Neutral: 0.760
#    Happy: 0.000
#    Sad: 0.238
#    Angry: 0.001
#    Fear: 0.000


# pip3 install --upgrade requests
sys.path.append(os.path.abspath(f"{____workingDir}/OpenVokaturi-4-0"))
sys.path.append(os.path.abspath(
    f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/"))
sys.path.append(os.path.abspath(
    f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/api/"))
import Vokaturi


print("Loading library...")
if platform.system() == "Darwin":
    assert struct.calcsize("P") == 8
    Vokaturi.load(os.path.abspath(
        f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/lib/open/__MACOSX/OpenVokaturi-4-0-mac.dylib"))
elif platform.system() == "Windows":
    if struct.calcsize("P") == 4:
        Vokaturi.load(os.path.abspath(
            f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/lib/open/win/OpenVokaturi-4-0-win32.dll"))
    else:
        assert struct.calcsize("P") == 8
        Vokaturi.load(os.path.abspath(
            f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/lib/open/win/OpenVokaturi-4-0-win64.dll"))
elif platform.system() == "Linux":
    assert struct.calcsize("P") == 8
    Vokaturi.load(os.path.abspath(
        f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/lib/open/linux/OpenVokaturi-4-0-linux.so"))
print("Analyzed by: %s" % Vokaturi.versionAndLicense())


def extractEmotionFromAudioFile(file_name: str):

    (sample_rate, samples) = scipy.io.wavfile.read(file_name)

    return extractEmotionFromAudioNdarray(sample_rate, samples)

def extractEmotionFromAudioBytes(audioBytes):

    (sample_rate, samples) = scipy.io.wavfile.read(io.BytesIO(audioBytes))

    return extractEmotionFromAudioNdarray(sample_rate, samples)


def extractEmotionFromAudioNdarray(sample_rate: int, samplesNdarray):

    print("   sample rate %.3f Hz" % sample_rate)
    print("Allocating Vokaturi sample array...")
    buffer_length = len(samplesNdarray)
    print("   %d samples, %d channels" % (buffer_length, samplesNdarray.ndim))
    c_buffer = Vokaturi.float64array(buffer_length)
    if samplesNdarray.ndim == 1:
        c_buffer[:] = samplesNdarray[:] / 32768.0  # mono
    else:
        c_buffer[:] = 0.5*(samplesNdarray[:, 0]+0.0 +
                           samplesNdarray[:, 1]) / 32768.0  # stereo

    print("Creating VokaturiVoice...")
    voice = Vokaturi.Voice(sample_rate, buffer_length, 0)

    print("Filling VokaturiVoice with samples...")
    voice.fill_float64array(buffer_length, c_buffer)

    print("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emotionProbabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotionProbabilities)

    temp = {
        "neutral": None,
        "happy": None,
        "sad": None,
        "angry": None,
        "fear": None
    }

    if quality.valid:
        print("Neutral: %.3f" % emotionProbabilities.neutrality)
        print("Happy: %.3f" % emotionProbabilities.happiness)
        print("Sad: %.3f" % emotionProbabilities.sadness)
        print("Angry: %.3f" % emotionProbabilities.anger)
        print("Fear: %.3f" % emotionProbabilities.fear)
        temp = {
            "neutral": emotionProbabilities.neutrality,
            "happy": emotionProbabilities.happiness,
            "sad": emotionProbabilities.sadness,
            "angry": emotionProbabilities.anger,
            "fear": emotionProbabilities.fear
        }
    else:
        print("Not enough sonorancy to determine emotions")

    voice.destroy()

    return temp


webApp = FastAPI()

folder_static = f"{____workingDir}/static"
isExist_static = os.path.exists(folder_static)
if not isExist_static:
    os.makedirs(folder_static)

webApp.mount(folder_static, StaticFiles(
    directory=folder_static), name="static")

webApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@webApp.get("/")
async def root():
    return "swagger API docs: /docs"


@webApp.post("/apis/audio/detect/emotion")
async def audioDetectEmotion(file: UploadFile = File(...)):
    speechBytes = await file.read()
    # audio_file_object = io.BytesIO()
    # sampleRate = 16000
    # with wave.open(audio_file_object, 'wb') as mem_file:
    #     mem_file.setnchannels(1)  # Mono audio
    #     mem_file.setsampwidth(2)  # 16-bit audio
    #     mem_file.setframerate(sampleRate)  # Sample rate
    #     mem_file.writeframes(speechBytes)
    # audio_file_object.seek(0)

    # audiobytes = audio_file_object.getvalue()
    # numpyData = numpy.frombuffer(audiobytes)
    
    return {
        "ok": 1,
        "data": extractEmotionFromAudioBytes(speechBytes)
    }


def runUvicorn(port):
    uvicorn.run(webApp, host="0.0.0.0", port=int(port), log_level="info")

# print("Reading sound file...")

# file_name = f"{____workingDir}/E_anhbd6_D_2023-01-04_H_085448_331_CLID_0971129816_210_21_NO.wav"
# file_name = f"{____workingDir}/2023-01-11-2022-0338954101-14.41.mp3"
# file_name = f"{____workingDir}/oh-yeah-everything-is-fine.wav"

# extractEmotionFromAudioFile(file_name)

if __name__ == "__main__":
    # _port=9981
    runUvicorn(9991)