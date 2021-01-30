FROM alpine:latest
COPY . /RNANet
WORKDIR /
RUN apk update && apk add --no-cache \
        curl \
        freetype-dev \
        gcc g++ \
        linux-headers \
        lapack-dev \
        make \
        musl-dev \
        openblas-dev \
        python3 python3-dev py3-pip py3-six py3-wheel \
        py3-matplotlib py3-requests py3-scipy py3-setproctitle py3-sqlalchemy py3-tqdm \
        sqlite \
    \
    && python3 -m pip install biopython pandas psutil pymysql && \
    \
    wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub && \
    wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.32-r0/glibc-2.32-r0.apk && \
    apk add glibc-2.32-r0.apk && \
    rm glibc-2.32-r0.apk && \
    \
    mkdir /3D && mkdir /sequences && \
    \
    mv /RNANet/scripts/x3dna-dssr /usr/local/bin/x3dna-dssr && chmod +x /usr/local/bin/x3dna-dssr && \
    \
    curl -SL http://eddylab.org/infernal/infernal-1.1.3.tar.gz | tar xz  && cd infernal-1.1.3 && \
    ./configure && make -j 16 && make install && cd easel && make install && cd / && \
    \
    curl -SL https://github.com/epruesse/SINA/releases/download/v1.7.1/sina-1.7.1-linux.tar.gz | tar xz && mv sina-1.7.1-linux /sina && \
    ln -s /sina/bin/sina /usr/local/bin/sina && \
    \
    rm -rf /infernal-1.1.3 && \
    \
    apk del openblas-dev gcc g++ gfortran binutils \
        curl \
        linux-headers \
        make \
        musl-dev \
        py3-pip py3-wheel \
        freetype-dev zlib-dev
VOLUME ["/3D", "/sequences", "/runDir"]
WORKDIR /runDir
ENTRYPOINT ["/RNANet/RNAnet.py", "--3d-folder", "/3D", "--seq-folder", "/sequences" ]