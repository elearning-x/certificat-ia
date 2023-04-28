# --------
# using scipy, it's kinda big but that should not be a problem
# base-notebook lacks at least numpy, widgets, so...
FROM nbhosting/scipy-notebook:latest

# --------
# install librosa, resampy and wavio
RUN true \
    && apt-get update \
    && apt-get install -y gcc portaudio19-dev python3-pyaudio \
    && pip install librosa \
    && pip install resampy \
    && pip install wavio \
    && pip install sounddevice \
    && true
