.DEFAULT_GOAL := qwen3moe
CXX        ?= g++
CXXFLAGS   ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS    ?=
NVCC       ?= nvcc
# RTX 3050 Laptop GPU (Ampere, compute capability 8.6); override for other GPUs.
NVCCFLAGS  ?= -O2 -std=c++17 -arch=sm_86
NVCC_CCBIN ?= /usr/bin/g++-15
CUDA_OBJ   :=
CUDA_TESTS :=
# GPU build: make CUDA=1 CUDA_HOME=/usr/local/cuda
# (switching CUDA on/off needs `make clean` first)
ifdef CUDA
CUDA_HOME ?= /usr/local/cuda
CXXFLAGS  += -DEAHMMF_CUDA -DEAHMMF_CUDA_COMPUTE -I$(CUDA_HOME)/include
LDFLAGS   += -L$(CUDA_HOME)/lib64 -lcudart
CUDA_OBJ  := expert_cuda.o
CUDA_TESTS := test_cuda_dp4a test_cuda_tensor_cores
endif

HDRS = q8k_activation_capture.h eahmmf.h etf.h eahmmf_optimizer.h unicode_tables.h pwea_client.h pwea_placement.h os_memory.h os_memory_monitor.h ram_budget_manager.h
PWEA_NATIVE_SRCS = pwea/native/src_pwea_unicode.cpp pwea/native/src_pwea_features.cpp pwea/native/src_pwea_model.cpp pwea/native/src_pwea_predictor.cpp
PWEA_NATIVE_HDRS = pwea/native/include/pwea_unicode.h pwea/native/include/pwea_unicode_tables.h pwea/native/include/pwea_features.h pwea/native/include/pwea_hash.h pwea/native/include/pwea_model.h pwea/native/include/pwea_predictor.h qwen3moe.cpp

expert_cuda.o: expert_cuda.cu
	$(NVCC) -ccbin $(NVCC_CCBIN) $(NVCCFLAGS) -c $< -o $@

qwen3moe: qwen3moe.cpp q6_avxvnni.h q6_groupwise_q8.h q6_groupwise_q8_simd.h q6_groupwise_q8_checked.h q6_groupwise_q8_v2.h eahmmf.h etf.h eahmmf_optimizer.h unicode_tables.h pwea_client.h pwea_placement.h os_memory.h os_memory_monitor.h ram_budget_manager.h $(PWEA_NATIVE_SRCS) $(PWEA_NATIVE_HDRS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -Ipwea/native/include -o $@ qwen3moe.cpp $(PWEA_NATIVE_SRCS) $(CUDA_OBJ) $(LDFLAGS)

test_pwea_placement: tests/test_pwea_placement.cpp $(HDRS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_pwea_placement.cpp $(CUDA_OBJ) $(LDFLAGS)

test_ram_budget_manager: tests/test_ram_budget_manager.cpp ram_budget_manager.h os_memory.h
	$(CXX) $(CXXFLAGS) -o $@ tests/test_ram_budget_manager.cpp $(LDFLAGS)

test_os_memory_monitor: tests/test_os_memory_monitor.cpp os_memory_monitor.h os_memory.h ram_budget_manager.h
	$(CXX) $(CXXFLAGS) -o $@ tests/test_os_memory_monitor.cpp $(LDFLAGS)

test_eahmmf: tests/test_eahmmf.cpp $(HDRS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_eahmmf.cpp $(CUDA_OBJ) $(LDFLAGS)

test_cuda_dp4a: tests/test_cuda_dp4a.cpp expert_cuda.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_cuda_tensor_cores: tests/test_cuda_tensor_cores.cpp expert_cuda.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cuda-test: $(CUDA_TESTS)
	@test -n "$(CUDA)" || { echo "cuda-test requires CUDA=1"; exit 2; }
	./test_cuda_dp4a
	./test_cuda_tensor_cores

unicode_tables.h: gen_unicode.py
	python3 gen_unicode.py

# full verification: numeric reference, pre-tokenizer, baseline equivalence, residency suite
check: qwen3moe test_eahmmf test_pwea_placement test_os_memory_monitor test_ram_budget_manager
	./test_os_memory_monitor
	./test_ram_budget_manager
	python3 tests/test_tiny.py
	python3 tests/test_pretok.py
	sh check_baseline.sh
	./test_eahmmf tiny.gguf
	./test_pwea_placement tiny.gguf

clean:
	rm -f qwen3moe test_eahmmf test_pwea_placement test_os_memory_monitor expert_cuda.o test_cuda_dp4a test_cuda_tensor_cores
