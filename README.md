3rd_pds_assigment

1.Dependencies
This project is written in CUDA C. In order to compile and run the code a machine with nvida toolkit is requiderd (nvcc, nvprof etc). Instructions on how to install and setup such an enviroment are available on the internet. In order to run CUDA the GPU must also support it.

2.Make
GNU Make is a tool which controls the generation of executables and other non-source files of a program from the program's source files. This is not strictly required but if you don't have it you must compile and run the project on your own.


3.Compile and run

On your Local computer

cd to the project directory and run one of the following Makefile targets:

make all: Runs all the targets this can take some time depending on the hardware.

make run_sequential: Compiles and runs a job for the sequential.c file.

make run_single_moment: Compiles and runs a job for the GPU_single_moment.cu file.

make run_block_moments: Compiles and runs a job for the GPU_block_moments.cu file.

make run_shared_memory: Compiles and runs a job for the GPU_shared_mem.cu file.



