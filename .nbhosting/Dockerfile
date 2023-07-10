# --------
# using scipy, it's kinda big but that should not be a problem
# base-notebook lacks at least numpy, widgets, so...
FROM nbhosting/scipy-notebook:latest

# --------
# install librosa, resampy and wavio
RUN true \
    && apt-get update \
    && pip install graphviz \
    && true
