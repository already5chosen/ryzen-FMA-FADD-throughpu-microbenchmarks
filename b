gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 128fma.c
gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 128fma1_add1.c
gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 128fma2_add1.c
gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 256fma.c
gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 256fma1_add1.c
gcc -c -Wall -O1 -mavx -mfma -mtune=haswell 256fma2_add1.c
g++ -c -Wall -O2 -mavx ryzen_fma_tests.cpp
g++ ryzen_fma_tests.o 128fma.o 128fma1_add1.o 128fma2_add1.o 256fma.o 256fma1_add1.o 256fma2_add1.o -o ryzen_fma_tests