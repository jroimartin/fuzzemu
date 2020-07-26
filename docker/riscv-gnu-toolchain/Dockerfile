# Build:
#   docker build \
#     [--build-arg USER_ID=<host_uid>] \
#     [--build-arg GROUP_ID=<host_gid>] \
#     [--build-arg USERNAME=<username>] \
#     -t riscv-gnu-toolchain .
# Run:
#   docker run \
#     -ti --rm \
#     -v $PWD:$PWD \
#     riscv-gnu-toolchain

FROM ubuntu:20.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=riscv

RUN export DEBIAN_FRONTEND='noninteractive' && \
    apt-get update && \
    apt-get install -y autoconf automake autotools-dev curl python3 \
        libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison \
        flex texinfo gperf libtool patchutils bc zlib1g-dev \
        libexpat-dev git && \
    apt-get clean && \
    unset DEBIAN_FRONTEND

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -u ${USER_ID} -g ${USERNAME} -m ${USERNAME}

RUN mkdir -p /opt/riscv && chown -R ${USERNAME}:${USERNAME} /opt/riscv

WORKDIR /home/${USERNAME}

USER ${USERNAME}

RUN git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
RUN cd riscv-gnu-toolchain && \
    ./configure --with-arch=rv64i --with-abi=lp64 --prefix=/opt/riscv && \
    make -j 16
