all: src/Makefile
	make -C src/ all

gdb: src/Makefile
	make -C src/ gdb

clean: src/Makefile
	make -C src/ clean

