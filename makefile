CC 			= gcc
DEBUG 		= -g
CFLAGS		= -c -Wall $(DEBUG) -O3
LFLAGS		= $(DEBUG)
DEPEND		= support.o dictman.o
CPUEXE		= cpu_md5
CPUOBJ		= $(DEPEND) cpu_md5.o cpu_main.o

NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
NVLD_FLAGS  = -lcudart -L/usr/local/cuda/lib64
GPUEXE		= cuda_md5
GPUOBJ		= cuda_main.o

$(CPUEXE): $(CPUOBJ)
	$(CC) $(CPUOBJ) -o $(CPUEXE)

cpu%.o: %.c
	$(CC) $(CFLAGS) -o $@ $<
	
cuda_main.o: cuda_main.cu kernel.cu support.cu support.h cuda_md5.h dictman.h
	$(NVCC) -c -o $@ cuda_main.cu $(NVCC_FLAGS)
		
$(GPUEXE): $(GPUOBJ)
	$(NVCC) $(GPUOBJ) -o $(GPUEXE) $(NVLD_FLAGS)
	
gpu: $(GPUEXE)

	cpu: $(CPUEXE)

default: cpu

clean:
	rm -rf *.o*
	rm -rf *.co
	rm -rf *.e*
