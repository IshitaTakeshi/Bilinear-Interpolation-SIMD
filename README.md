## Bilinear interpolation using SIMD

This repository contains C implementation of bilinear interpolation and the Python wrapper of it

## Requirement

CPU with AVX2 support

## C implementation

### How to run
```
$ clang -g -Wall -mavx -mavx2 _bilinear.c -o bilinear
$ ./bilinear
```

## Python wraper
### Requirements

```
cython, numpy, scipy
```

### How to build

```
python3 setup.py build_ext --inplace
```

### How to run

```
python3 main.py
```
