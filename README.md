## Bilinear interpolation using SIMD

This repository contains C implementation of bilinear interpolation and the Python wrapper of it

### Requirement

CPU with AVX2 support

### C implementation

#### How to run

```
$ clang -g -Wall -mavx -mavx2 _bilinear.c _print.c main.c -o bilinear
$ ./bilinear
```
#### Benchmark

```
$ clang -Wall -Ofast -mavx -mavx2 _bilinear.c benchmark.c -o benchmark
$ ./benchmark
```

### Python wrapper
#### Requirements

For main

```
cython, numpy, scipy
```

For benchmark

```
cython, numpy, numba, scikit-image
```

#### How to build

```
$ python3 setup.py build_ext --inplace
```

#### How to run

```
$ python3 main.py
```

#### Benchmark

```
$ python3 benchmark.py
```
