# -*-makefile-*-

# This file is based on the Makefile that is included in the "egstra" package:
# http://groups.csail.mit.edu/nlp/egstra
#
# At some point, I intend to write scripts that will use the GNU build system
# (to check for dependencies and whatnot), but for now, this works for me.

# The current dependencies are:
#   Boost (http://www.boost.org), version 1.48 or higher.
#   LBFGS (http://www.chokkan.org/software/liblbfgs), version 1.10 or higher.
#   QuadProg++ (http://sourceforge.net/projects/quadprog), version 1.2 or higher.
#   -- Note: Latent Struct uses the ublas-based library, uQuadProgpp.
#
# If any of these libraries are installed in non-standard locations, you will
# need to add a -I<path to header(s)> entry to INCLUDES, and a -L<path to lib(s)>
# entry to LIBS below.

INCLUDES = -I. -Ibase -Idomain_common -Idomain_sentences -Idomain_synthetic -Idomain_words -Iedit_operations -Igraph -Ihypergraph -Imodels -Ioptimization -Itrain_objectives -Iutil -I/usr/local/include
LIBS = -L/usr/local/lib -ldl -lm -llbfgs -luQuadProgpp -lboost_program_options -lboost_timer -lboost_system -lboost_thread -lboost_regex -lboost_filesystem
CCFLAGS = -frounding-math -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CC = g++ -O3 -DNDEBUG $(CCFLAGS) -g
CCDEP = $(CC) $(INCLUDES) -MM
CCOBJ = $(CC) $(INCLUDES) -c
CCGDB = g++ -O0 $(CCFLAGS)
CCGDBOBJ = $(CCGDB) $(INCLUDES) -ggdb -c

SRCS = $(shell find . -name "*.cpp")
OBJS = $(patsubst %.cpp,%.o,$(SRCS))
DEPS = $(patsubst %.cpp,%.dep,$(SRCS))
GDBOBJS = $(patsubst %.cpp,%-gdb.o,$(SRCS))

EXEC = ../latent_struct ../latent_struct-gdb

all : ../latent_struct

gdb : ../latent_struct-gdb

../latent_struct : obj
	$(CC) $(OBJS) $(LIBS) -o $@

../latent_struct-gdb : obj-gdb
	$(CCGDB) $(GDBOBJS) $(LIBS) -o $@

dep : $(DEPS)

obj : dep
	@$(MAKE) NESTED=true -s obj-internal

obj-gdb : dep
	@$(MAKE) NESTED=true -s obj-gdb-internal

%.dep : %.cpp
	@echo DEP $<
	-@rm -f $@
	@echo -n $(<D)/ > $@
	@$(CCDEP) $< >> $@
	@echo '	@echo CC $$@' >> $@
	@echo '	@$$(CCOBJ) $$< -o $$@' >> $@
	@echo "" >> $@
	@$(CCDEP) -MT $*-gdb.o $< >> $@
	@echo '	@echo CC-gdb $$@' >> $@
	@echo '	@$$(CCGDBOBJ) $$< -o $$@' >> $@

clean :
	-find . -name \*.gch -exec echo rm {} \; -exec rm {} \;
	-find . -name \*.dep -exec echo rm {} \; -exec rm {} \;
	-find . -name \*.o -exec echo rm {} \; -exec rm {} \;
	-find . -name \*~ -exec echo rm {} \; -exec rm {} \;
	-rm -f $(EXEC) $(EXECTEST)


# these rules are only active from a nested invocation of make (see
# above)
ifdef NESTED
include $(DEPS)

obj-internal : $(OBJS)

obj-gdb-internal : $(GDBOBJS)

endif


.PHONY : clean all gdb dep obj obj-gdb obj-internal obj-gdb-internal
