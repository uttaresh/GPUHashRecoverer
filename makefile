CC 			= gcc
DEBUG 		= -g
CFLAGS		= -c -Wall $(DEBUG) -O3
LFLAGS		= $(DEBUG)
DEPEND		= support.o
CPUEXE		= cpu_md5
CPUOBJ		= $(DEPEND) cpu_md5.o cpu_main.o dictman.o brute_force.o

NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20 -gencode arch=compute_20,code=sm_20
NVLD_FLAGS  = -lcudart -L/usr/local/cuda/lib64
GPUEXE		= cuda_md5
GPUOBJ		= cuda_main.o

$(CPUEXE): $(CPUOBJ)
	$(CC) $(CPUOBJ) -o $(CPUEXE)

cpu%.o: %.c
	$(CC) $(CFLAGS) -o $@ $<
	
dictman.o: dictman.c dictman.h
	$(CC) $(CFLAGS) -o $@ dictman.c
	
brute_force.o: brute_force.c brute_force.h
	$(CC) $(CFLAGS) -o $@ brute_force.c
	
cuda_main.o: cuda_main.cu kernel.cu support.cu support.h cuda_md5.h
	$(NVCC) -c -o $@ cuda_main.cu $(NVCC_FLAGS)
		
$(GPUEXE): $(GPUOBJ)
	$(NVCC) $(GPUOBJ) -o $(GPUEXE) $(NVLD_FLAGS)
	
gpu: $(GPUEXE)

cpu: $(CPUEXE)

md5: md5_wiki.c
	gcc -o md5 md5_wiki.c

default: cpu gpu md5

cleangem:
	rm -rf *.o*
	rm -rf *.e*

clean:
	rm -rf *.o
	rm -rf $(CPUEXE)
	rm -rf $(GPUEXE)
