import os

from cffi import FFI

os.system(
    "gcc -shared src/farthest_point_sampling.cpp -c -o src/farthest_point_sampling.cpp.o -fopenmp -fPIC -O2 -std=c++11"
)

ffibuilder = FFI()

with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=["src/farthest_point_sampling.cpp.o"],
    libraries=["stdc++"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    os.system("rm *.o")
