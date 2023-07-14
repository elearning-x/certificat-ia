# --------
# using scipy, it's kinda big but that should not be a problem
# base-notebook lacks at least numpy, widgets, so...
FROM nbhosting/scipy-notebook:latest

# --------

RUN true \
    && sudo apt-get update \
    && sudo apt-get install -y graphviz \
    && pip install graphviz \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && true
