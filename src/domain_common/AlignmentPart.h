/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "StateType.h"
#include <string>

#ifndef _ALIGNMENTPART_H
#define _ALIGNMENTPART_H

typedef struct alignment_part {

  const StateType* state;
  
  std::string source;
  
  std::string target;
  
} AlignmentPart;

#endif
