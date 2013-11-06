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

#include "Label.h"
#include "Pattern.h"
#include "Ublas.h"
#include <boost/shared_array.hpp>

// A data structure that stores a vector of real values, and optionally the
// value of a latent (i.e., unobserved during training) variable z.
class SparsePattern : public Pattern {

  public:
    SparsePattern(boost::shared_array<double> values, std::size_t len,
        Label z = -1) : _z(z) {
      _vector = SparseRealVec(len);
      for (std::size_t i = 0; i < len; ++i) {
        if (values[i] != 0)
          _vector(i) = values[i];
      }
    }
    
    SparsePattern(const SparseRealVec& vector, Label z = -1) : _z(z) {
      _vector = vector;
    }

    const SparseRealVec& getVector() const {
      return _vector;
    }
    
    Label getZ() {
      return _z;
    }

    virtual int getSize() const {
      return _vector.size();
    }
    
    virtual std::string getHashString() const {
      assert(0); // not implemented
      return "";
    }

  private:
    SparseRealVec _vector;
    Label _z;
};

#endif
