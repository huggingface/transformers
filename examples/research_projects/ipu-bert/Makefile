all: custom_ops/plugin_version custom_ops.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

custom_ops.so: custom_ops/plugin_version custom_ops/*.cpp custom_ops/workarounds/*.cpp
	g++ -std=c++14 -fPIC \
		-DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
		-DONNX_NAMESPACE=onnx \
		custom_ops/params.cpp \
		custom_ops/sparse_accumulate.cpp \
		custom_ops/sparse_accumulate_pattern.cpp \
		custom_ops/tied_gather.cpp \
		custom_ops/tied_gather_pattern.cpp \
		custom_ops/outline_dropout_pattern.cpp \
		custom_ops/seed_modify.cpp \
		custom_ops/lamb_serialised_weight_pattern.cpp \
		custom_ops/one_hot.cpp \
		custom_ops/workarounds/accumulate_priority_pattern.cpp \
		custom_ops/workarounds/stash_layout_pattern.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so custom_ops/plugin_version 
