# riscv-emu

This project is just me trying to replicate the work done by **Brandon Falk**
during the **Fuzz Week** for learning purposes and experimentation.

## Fuzz Week

- [Blog post](https://gamozolabs.github.io/2020/07/12/fuzz_week_2020.html)
- [GitHub repository](https://github.com/gamozolabs/fuzz_with_emus/)
- [YouTube playlist](https://www.youtube.com/playlist?list=PLSkhUfcCXvqHsOy2VUxuoAf5m_7c8RqvO)

## Build riscv-gnu-toolchain

```
$ sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev
$ git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
$ cd riscv-gnu-toolchain
$ ./configure --with-arch=rv64i --with-abi=lp64 --prefix=<PREFIX>
$ make -j <NJOBS>
```

## Profiling

### Perf

```
$ sudo perf record --call-graph=dwarf ./target/release/riscv-emu
$ sudo perf report --hierarchy -M intel
```

Alternatively:

```
$ sudo perf top
```

### Valgrind

```
$ valgrind --tool=callgrind ./target/release/riscv-emu
$ callgrind_annotate --tree=both --inclusive=yes callgrind.out.<pid>  # or KCachegrind
```
