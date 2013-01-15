/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SPARSEPATTERN_H
#define _SPARSEPATTERN_H

#include "Pattern.h"
#include "Ublas.h"
#include <boost/shared_array.hpp>

// A data structure that stores a vector of real values.
class SparsePattern : public Pattern {

  public:
    SparsePattern(boost::shared_array<double> values, std::size_t len) {
      _vector = SparseRealVec(len);
      for (std::size_t i = 0; i < len; ++i) {
        if (values[i] != 0)
          _vector(i) = values[i];
      }
    }
    
    SparsePattern(const SparseRealVec& vector) {
      _vector = vector;
    }

    const SparseRealVec& getVector() const {
      return _vector;
    }

    virtual int getSize() const {
      return _vector.size();
    }

  private:
    SparseRealVec _vector;
};

#endif
