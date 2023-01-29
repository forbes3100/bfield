
all: bfield testmath

bfield: bfield_main.cpp bfield_server.cpp bfield_server.h
	gcc -O2 bfield_main.cpp bfield_server.cpp -std=c++11 -lm -lkqueue -lstdc++ -lpng -o bfield

testmath: testmath.cpp bfield_server.cpp bfield_server.h
	gcc -O2 testmath.cpp bfield_server.cpp -std=c++11 -lm -lstdc++ -lpng -o testmath

distclean:
	rm -rf bfield testmath cache_dp __pycache__
