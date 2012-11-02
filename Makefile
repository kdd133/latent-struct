all: src/Makefile
	+make -C src/ all

gdb: src/Makefile
	+make -C src/ gdb

clean: src/Makefile
	make -C src/ clean
	make -C tests/ clean

check: tests/Makefile gdb
	+make -C tests/ all
