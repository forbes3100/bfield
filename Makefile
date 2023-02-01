
all: bfield testmath

here = $(shell pwd)

bfield: bfield_main.cpp bfield_server.cpp bfield_server.h
	gcc -O2 bfield_main.cpp bfield_server.cpp -std=c++11 -lm -lkqueue -lstdc++ -lpng -o bfield

testmath: testmath.cpp bfield_server.cpp bfield_server.h
	gcc -O2 testmath.cpp bfield_server.cpp -std=c++11 -lm -lstdc++ -lpng -o testmath

test_blend: test_blend.py
	./bl -b -P $(here)/bfield.py -P $(here)/test_blend.py

distclean:
	rm -rf bfield testmath cache_dp __pycache__
