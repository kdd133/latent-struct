/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SYNTHETIC_DATA_H
#define _SYNTHETIC_DATA_H

#include "Dataset.h"
#include "Parameters.h"
#include "Ublas.h"

class SyntheticData {

  public:
    static void generate(std::size_t m, std::size_t nx, std::size_t ny,
        std::size_t nz, const Parameters& theta, Dataset& dataset, int seed = 0);
      
    static SparseRealVec prob_x(const Parameters& theta, const SparsePattern& x,
        std::size_t ny, std::size_t nz);
};

#endif
