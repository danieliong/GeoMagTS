LIB_DIR = glmgen_c
OBJ = glmgen.cpython-37m-darwin.so
SETUP = setup_glmgen.py
PYX = glmgen.pyx

default: glmgen

all: glmgen_c glmgen 

glmgen: $(SETUP) $(PYX)
	python $(SETUP) build_ext --inplace 
	rm -rf build/ glmgen.c
	install_name_tool -change lib/libglmgen.so $(LIB_DIR)/lib/libglmgen.so $(OBJ)

glmgen_c:
	make -C $(LIB_DIR) all

clean:
	rm $(OBJ)

cleanall:
	clean
	make -C $(LIB_DIR) cleanall