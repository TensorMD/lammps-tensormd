#CC = mpicc
CC = swgcc
LD = swgcc
CXX = swg++
EXE=kernel1_main kernel2_main kernel3_main kernel4_main nnp_main

CFLAGS = -msimd -mieee -DCHECK_RESULT
LDFLAGS = -mhybrid $(CFLAS)

INCLUDES = -I../include/ -I/usr/sw/yyzlib/lapack-3.8.0/CBLAS/include
EXE_INCLUDES = -I/gfs_data/home/swyyz/swz/reflib/lapack-3.8.09a/include $(INCLUDES)

EXE_LIBS = -L/gfs_data/home/swyyz/swz/reflib/lapack-3.8.09a -L/usr/sw/yyzlib/lapack-3.8.0 -lcblas -L/gfs_data/home/swyyz/swz/reflib/xMath-SACA -L/usr/sw/yyzlib/xMath-SACA -lswblas -lgfortran -lz

OBJDIR = ./obj

slavesrc=$(wildcard ./slave/*.c)

mastersrc=$(wildcard ./master/*.c)

LIBOBJ1 += $(patsubst %.c, %.o, $(slavesrc) $(mastersrc))

LIBOBJ2 = $(notdir $(LIBOBJ1))

LIBOBJ3 += $(patsubst %.o, obj/%.o, $(LIBOBJ2))

OBJ = ${LIBOBJ3}

LIB = ../lib/libswTensorMD.a
#LIB = ./libswdgemm.so

all:$(EXE)

ar:$(LIB)

example:$(EXE)
 
kernel1_main: kernel1_main.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $^ $(EXE_LIBS) -lm -lm_slave

kernel2_main: kernel2_main.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $^ $(EXE_LIBS) -lm

kernel3_main:kernel3_main.o calculate_dM.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $^ $(EXE_LIBS) -lm

kernel4_main:kernel4_main.o utils.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $^ $(EXE_LIBS) -lm

nnp_main: nnp_main.o nnp.o cnpy.o $(LIB)
	$(CXX) $(LDFLAGS) -o $@ $^ $(EXE_LIBS) -lm -lm_slave

kernel1_main.o:kernel1_main.c
	$(CC) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

kernel2_main.o:kernel2_main.c
	$(CC) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

kernel3_main.o:kernel3_main.c
	$(CC) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

calculate_dM.o:calculate_dM.c
	$(CC) $(CFLAGS) $(INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

kernel4_main.o:kernel4_main.c
	$(CC) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

nnp_main.o:nnp_main.cpp
	$(CXX) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

nnp.o:nnp.cpp
	$(CXX) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

cnpy.o:cnpy.cpp
	$(CXX) $(CFLAGS) $(EXE_INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

utils.o:utils.c
	$(CC) $(CFLAGS) $(INCLUDES) -O2 -mhost -faddress_align=128 -c $^ -o $@

$(LIB): $(OBJ)
	#sw9gcc -mdynamic -shared -o  $@ $^
	#swar rcs libswTensorMD.a obj/*.o
	swar rcs $@ $^ 

obj/%.o: ./master/%.c
	#$(CC) $(CFLAGS) -O2 -mhost  -fPIC -c $^ -o $@
	$(CC) $(CFLAGS) $(INCLUDES) -O2 -mhost -c $^ -o $@

obj/%.o: ./slave/%.c
	#$(CC) $(CFLAGS) -O2 -mslave -fPIC -faddress_align=256 -Dperf -c $^ -o $@
	$(CC) $(CFLAGS) $(INCLUDES) -O2 -mslave -faddress_align=256 -Dperf -c $^ -o $@

.PHONY: clean
clean:
	rm -rf $(EXE) *.o obj/*.o


