# OpenVokaturiFastApi

download from

                https://vokaturi.com/downloads/download-the-sdk


extract to folder eg:

                /program.py
                /OpenVokaturi-4-0/OpenVokaturi-4-0/api/Vokaturi.py
                /OpenVokaturi-4-0/OpenVokaturi-4-0/lib/...


# Run to test

                python3 program.py 9988

# Run to generate jwt_as secret 

                python3 program.py jwt

# Docker 

Can use env for docker: APP_KEY , SECRET_KEY, JWT_KEY_AS_SECRET

                docker build -f dockerfile -t vocal-emotion-detector .

                docker run -d --restart always -p 9988:9988 --name vocal-emotion-detector_9988 vocal-emotion-detector