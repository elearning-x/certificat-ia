# --------
# using scipy, it's kinda big but that should not be a problem
# base-notebook lacks at least numpy, widgets, so...
FROM nbhosting/scipy-notebook:latest

# --------
# install librosa, resampy and wavio
RUN true \
    && apt-get update \
    && apt-get install -y graphviz \
    && pip install graphviz \
    && pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && true
