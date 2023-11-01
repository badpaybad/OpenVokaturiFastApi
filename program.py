# https://vokaturi.com/using-vokaturi/example-code-for-python-batch

import io
import wave
import numpy
from pydantic import ValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer,OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
import hashlib
import uvicorn
#from jwcrypto import jwt, jwk
import struct
import platform

import os
import datetime
import math
import sys
import threading
import re

from typing import Optional
import sys
import scipy.io.wavfile

from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt 

from typing import Union


SECURITY_ALGORITHM = 'HS256'
APP_KEY =os.getenv("APP_KEY", '211284dd-1e4e-4caf-ba37-8a8a0b211302')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#generate key by commandline: openssl rand -hex 32
SECRET_KEY=os.getenv("SECRET_KEY","09d25e021faa6ca2126c818184b7a9563b93f7099f6f0f02aa13f63bdd841221")
JWT_KEY_AS_SECRET=os.getenv("JWT_KEY_AS_SECRET","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzdGF0aWMiLCJpZCI6IjU5YmFiZjgwZjIwZmZmYjk4NjUxYTNlNDk1OGZiMzg3YzM2NzVmMmQ0NjI2MWE1MjM3YWU4NjEzNjQ5NTk5MzAiLCJleHAiOjQ4NTI0NDY2ODB9.JPrA_K7YFnJ3j9zUHnrBxxmXUW8ARs52PHugab4xlpU")

ACCESS_TOKEN_EXPIRE_MINUTES=60*24*365*100 #100 nam sau 

_http_port = str(sys.argv[1])

# insert at 1, 0 is the script path (or '' in REPL)
____workingDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, ____workingDir)

print("____workingDir", ____workingDir)

#jwt auth https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/#__tabbed_1_3
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)
def get_password_hash(password):
    return pwd_context.hash(password)

def hash256hex(text):
    m = hashlib.sha256(text.encode('UTF-8'))
    return m.hexdigest()

def create_jwt_token(data: dict, expires_delta_timedelta=None):
    to_encode = data.copy()
    if expires_delta_timedelta!=None:
        expire = datetime.utcnow() + expires_delta_timedelta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=SECURITY_ALGORITHM)
    return encoded_jwt

#print(create_jwt_token({"sub":"static","id":hash256hex(APP_KEY)}))


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

sys.path.append(os.path.abspath(f"{____workingDir}/OpenVokaturi-4-0"))
sys.path.append(os.path.abspath(
    f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/"))
sys.path.append(os.path.abspath(
    f"{____workingDir}/OpenVokaturi-4-0/OpenVokaturi-4-0/api/"))
import Vokaturi

class AppContext:
    def __init__(self) :
        self.apppkeyhashhex=hash256hex(APP_KEY)
        pass
    
appContext=AppContext()

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

import uuid
import subprocess
def convertToWavFromBytes(audioBytes, fileNameInput:str, sourceChannel:int=0):
    
    fileExt=fileNameInput.split(".")[-1]+""
    
    print(f"fileExt: {fileExt} fileNameInput: {fileNameInput}")
    ac=2
    if fileExt.lower()=="wav":
        if sourceChannel <= 0:
            return audioBytes
        else:
            ac = sourceChannel
    else:
        if sourceChannel > 0:
            ac = sourceChannel
            
    # with wave.open(abs_path_file, 'wb') as wav_file:
    #     wav_file.setnchannels(1) # Mono audio
    #     wav_file.setframerate(16000)
    #     wav_file.setsampwidth(2)# 16-bit audio
    #     wav_file.writeframes(audioBytes)
    uinqueId=f"{datetime.datetime.now().timestamp()}.{uuid.uuid4()}"
    pathInputFile=f"{____workingDir}/static/{uinqueId}.{fileNameInput}"
  
    with open(pathInputFile, 'wb') as f:
        f.write(audioBytes)
        
    wavAbsPathFile= f"{____workingDir}/static/{uinqueId}.{fileNameInput}.wav"
    # convert mp3 to wav file, all to wav as stereo
    subprocess.call(['ffmpeg','-y', '-i', pathInputFile,'-ac',f'{ac}', wavAbsPathFile])
    
    temp=[]
    with open(wavAbsPathFile, "rb") as fr:
        temp=fr.read()
    
    print(f"Converted into: {wavAbsPathFile}")    
    
    os.remove(pathInputFile)
    os.remove(wavAbsPathFile)
    return temp

def extractEmotionFromAudioFile(file_name: str):

    (sample_rate, samples) = scipy.io.wavfile.read(file_name)
    
    print(f"extractEmotionFromAudioFile rate: {sample_rate}")

    return extractEmotionFromAudioNdarray(sample_rate, samples)

def extractEmotionFromAudioBytes(audioBytes):

    (sample_rate, samples) = scipy.io.wavfile.read(io.BytesIO(audioBytes))
    
    #mono_data = numpy.mean(samples, axis=1, dtype=numpy.int16)
    
    print(f"extractEmotionFromAudioBytes rate: {sample_rate}")

    return extractEmotionFromAudioNdarray(sample_rate, samples)

def extractEmotionFromAudioNdarray(sample_rate: float, samplesNdarray):

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

    if quality.valid:
        # print("Neutral: %.3f" % emotionProbabilities.neutrality)
        # print("Happy: %.3f" % emotionProbabilities.happiness)
        # print("Sad: %.3f" % emotionProbabilities.sadness)
        # print("Angry: %.3f" % emotionProbabilities.anger)
        # print("Fear: %.3f" % emotionProbabilities.fear)
        temp = {           
                "neutral":round ( emotionProbabilities.neutrality,3),
                "happy": round(emotionProbabilities.happiness,3),
                "sad": round(emotionProbabilities.sadness,3),
                "angry": round( emotionProbabilities.anger,3),
                "fear": round( emotionProbabilities.fear,3),
                "msg":None
        }
    else:
        print("Not enough sonorancy to determine emotions")
        temp = {
        "neutral": None,
        "happy": None,
        "sad": None,
        "angry": None,
        "fear": None,
        "msg":"Not enough sonorancy to determine emotions"
    }

    voice.destroy()

    return (temp, sample_rate,samplesNdarray)

def extractEmotionFromAudioNdarrayInterval(sample_rate: float, samplesNdarray, timeStep:float=2):
    
    print("   sample rate %.3f Hz" % sample_rate)
    print("Allocating Vokaturi sample array...")
    numberOfSamples = len(samplesNdarray)
    print("   %d samples, %d channels" % (numberOfSamples, samplesNdarray.ndim))
    duration = numberOfSamples / sample_rate
        
    bufferSafetyTime = 1.0
    bufferDuration = timeStep + bufferSafetyTime
    bufferLength = int(sample_rate * bufferDuration)
    if(bufferLength> numberOfSamples):
        bufferLength= numberOfSamples
        
    numberOfSteps = int(duration / timeStep )
    if numberOfSteps<=0:
        numberOfSteps=1
    
    print(f"duration: {duration}; numberOfSteps: {numberOfSteps}; bufferLength: {bufferLength}")
    
    voice = Vokaturi.Voice (  sample_rate, bufferLength, 0  )
    
    dicResult=[]
    
    for istep in range(0,numberOfSteps):        
        startingTime = istep * timeStep
        startingSample =int( startingTime * sample_rate)
        
        nstep=istep+1
        endTime = nstep * timeStep       
        if istep== numberOfSteps-1:
            endTime= duration        
        endSample =int( endTime * sample_rate )     
        if (endSample > numberOfSamples):
            endSample = numberOfSamples
        
        #print(f"startingTime -> endTime: {startingTime} -> {endTime} at istep: {istep}/{numberOfSteps}")
        #print(f"startingSample -> endSample: {startingSample} -> {endSample} at istep: {istep}/{numberOfSamples}")
        c_buffer = Vokaturi.float64array(int(endSample-startingSample))
        if samplesNdarray.ndim == 1:
            c_buffer[:] = samplesNdarray[startingSample:endSample] / 32768.0  # mono
        else:
            c_buffer[:] = 0.5*(samplesNdarray[startingSample:endSample, 0]+0.0 +
                            samplesNdarray[startingSample:endSample, 1]) / 32768.0  # stereo
            
        
        voice.fill_float64array( int(endSample-startingSample)   , c_buffer)
        
        quality = Vokaturi.Quality()
        
        emotionProbabilities = Vokaturi.EmotionProbabilities()
        voice.extract(quality, emotionProbabilities)
      
      
        if quality.valid:
            # print("Neutral: %.3f" % emotionProbabilities.neutrality)
            # print("Happy: %.3f" % emotionProbabilities.happiness)
            # print("Sad: %.3f" % emotionProbabilities.sadness)
            # print("Angry: %.3f" % emotionProbabilities.anger)
            # print("Fear: %.3f" % emotionProbabilities.fear)
            temp = {
                "neutral":round ( emotionProbabilities.neutrality,3),
                "happy": round(emotionProbabilities.happiness,3),
                "sad": round(emotionProbabilities.sadness,3),
                "angry": round( emotionProbabilities.anger,3),
                "fear": round( emotionProbabilities.fear,3),
                "msg":None
            }
        else:
            print("Not enough sonorancy to determine emotions")
            temp = {
                "neutral": None,
                "happy": None,
                "sad": None,
                "angry": None,
                "fear": None,
                "msg":"Not enough sonorancy to determine emotions"
            }            
            
        dicResult.append (
            {
            "startingTime":startingTime,
            "endTime":endTime,
            "data":temp,            
        }
        )
    voice.destroy()
    return (dicResult, sample_rate, samplesNdarray)
    pass

webApp = FastAPI()

reusable_oauth2 = HTTPBearer(
    scheme_name='Authorization'
)

#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# @webApp.middleware("http")
# async def jwt_middleware_authenticate(request: Request, call_next):
#     #start_time = time.time() Bearer
#     authToken=None

#     #https://jwcrypto.readthedocs.io/en/latest/

#     if  "authorization" in request.headers:
#         authToken= request.headers["authorization"].strip()
#     elif "Authorization" in request.headers:
#         authToken= request.headers["Authorization"].strip()

#     if authToken==None or authToken=="":
#         #raise HTTPException(status_code=401, detail="Unauthenticate")
#         return Response("Unauthenticate", status_code=401)

#     if authToken.startswith("Bearer") or authToken.startswith("bearer"):
#         authToken=authToken[6:].strip()

#     k = {"k": APP_KEY, "kty": "oct"}
#     key = jwk.JWK(**k)

#     jwt.JWT(key=key, jwt=authToken)

#     payload =  jwt.JWT(key=key, jwt=authToken)

#     if datetime.datetime(payload.get('exp')) < datetime.datetime.now():
#         raise Response( "Token expired",status_code=403)

#     request.state.jwt=payload #send token 
#     response = await call_next(request)
#     # process_time = time.time() - start_time
#     # response.headers["X-Process-Time"] = str(process_time)
#     return response
from typing_extensions import Annotated
async def jwt_validate(request: Request,token= Depends(reusable_oauth2)):
#async def jwt_validate(request: Request,token= Depends(oauth2_scheme)):
    try:
        authToken=""
        print(f"token: {token}")
        
        if  "authorization" in request.headers:
            authToken= request.headers["authorization"].strip()
        elif "Authorization" in request.headers:
            authToken= request.headers["Authorization"].strip()

        if authToken==None or authToken=="":
            raise HTTPException(status_code=401, detail="Unauthenticate")
            
        if authToken.startswith("Bearer") or authToken.startswith("bearer"):
            authToken=authToken[6:].strip()
                
        payload = jwt.decode(authToken, SECRET_KEY, algorithms=[SECURITY_ALGORITHM])
        
        appkey_hashed_claim = payload.get("id")
            
        if(appkey_hashed_claim!=appContext.apppkeyhashhex):
            raise HTTPException(status_code=401, detail="Unauthenticate")
        
        request.state.jwt=payload #send token 
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=401, detail="Unauthenticate")
        pass

@webApp.get("/apis/auth/test",dependencies=[Depends(jwt_validate)])
##@webApp.get("/apis/auth/test")
async def TestAuth(request:Request):

    return request.state.jwt

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


@webApp.post("/apis/audio/detect/emotion",dependencies=[Depends(jwt_validate)])
async def audioDetectEmotion(file: UploadFile = File(...),convertToAudioChannel:int=Form(0), stepInSeconds:int=Form(0)):
#async def audioDetectEmotion(file: UploadFile = File(...),audioChannel:Optional[int]=2, stepInSeconds:Optional[int]=2):
    try:
        speechBytes = await file.read()
        
        fileNameInput= file.filename
        
        speechBytes= convertToWavFromBytes(speechBytes,fileNameInput,convertToAudioChannel)
        
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
        full,sample_rate, samples= extractEmotionFromAudioBytes(speechBytes)
        chunks=[]
        if stepInSeconds==None or stepInSeconds<=0:
            pass
        else:
            chunks, sample_rate,samples = extractEmotionFromAudioNdarrayInterval(sample_rate, samples,stepInSeconds)
            pass
        
        return {
            "ok": 1,
            "full":full,
            "chunks": chunks,
            "file":fileNameInput,
            "err":None
        }
    except Exception as ex:
        print(f"audioDetectEmotion#ERR: {ex}")
        return {
            "ok": 0,
            "full":None,
            "chunks": None,
            "file":fileNameInput,
            "err":ex
        }


def runUvicorn(port):
    uvicorn.run(webApp, host="0.0.0.0", port=int(port), log_level="info")

# print("Reading sound file...")

# file_name = f"{____workingDir}/E_anhbd6_D_2023-01-04_H_085448_331_CLID_0971129816_210_21_NO.wav"
# file_name = f"{____workingDir}/2023-01-11-2022-0338954101-14.41.mp3"
file_name = f"{____workingDir}/oh-yeah-everything-is-fine.wav"
#file_name=f"{____workingDir}/1690830159.054426.e1c9f763-7b55-44ba-b609-15d7cbd42c85.2023-01-11-2022-0338954101-14.41.mp3.wav"

full= extractEmotionFromAudioFile(file_name)
print(full)
(sample_rate, samples) = scipy.io.wavfile.read(file_name)
dicres= extractEmotionFromAudioNdarrayInterval(sample_rate, samples, 30)

print(dicres)

if __name__ == "__main__":    
    runUvicorn(_http_port)
