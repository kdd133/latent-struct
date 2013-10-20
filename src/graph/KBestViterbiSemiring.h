/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _KBESTVITERBISEMIRING_H
#define _KBESTVITERBISEMIRING_H

#include "Hyperedge.h"
#include "LogWeight.h"
#include "ViterbiSemiring.h"
#include <algorithm>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <vector>

class KBestViterbiSemiring {

public:
  KBestViterbiSemiring() : _size(0) { }

  KBestViterbiSemiring(LogWeight score) : _size(1),
      _entries(new ViterbiSemiring[1]) {
    _entries[0] = ViterbiSemiring(score);
  }
  
  KBestViterbiSemiring(const Hyperedge& edge) : _size(1),
      _entries(new ViterbiSemiring[1]) {
    _entries[0] = ViterbiSemiring(edge);
  }
  
  // Take two sorted vectors of (up to) length k as input, and output the top
  // k in sorted order.
  KBestViterbiSemiring& operator+=(const KBestViterbiSemiring& rhs) {
    const int most = _size + rhs._size; // at most this many outputs (up to 2k)
    boost::scoped_array<ViterbiSemiring> newEntries(new ViterbiSemiring[most]);
    
    // Merge the two sorted vectors (i.e., the merge step of merge sort).
    int count = 0, i = 0, j = 0;
    while (i < _size && j < rhs._size) {
      if (!_entries[i].bp())
        i++;
      else if (!rhs._entries[j].bp())
        j++;
      else if (_entries[i] < rhs._entries[j])
        newEntries[count++] = _entries[i++];
      else
        newEntries[count++] = rhs._entries[j++];
    }
    for (; i < _size; i++) {
      if (_entries[i].bp())
        newEntries[count++] = _entries[i];
    }
    for (; j < rhs._size; j++) {
      if (rhs._entries[j].bp())
        newEntries[count++] = rhs._entries[j];
    }
    
    // Return the top k entries (or all the entries if fewer than k).
    _size = std::min(count, k);
    _entries.reset(new ViterbiSemiring[_size]);
    for (int i = 0; i < _size; i++)
      _entries[i] = newEntries[i];
    return (*this);
  }
  
  KBestViterbiSemiring& operator*=(const KBestViterbiSemiring& rhs) {
    // General case: Enumerate the possible derivations.
    const int total = _size * rhs._size;
    std::vector<ViterbiSemiring> temp(total);
    int next = 0;
    for (int i = 0; i < _size; i++)
      for (int j = 0; j < rhs._size; j++) {
        temp[next] = _entries[i];
        temp[next] *= rhs._entries[j];
        next++;
      }
      
    // Sort the derivations by weight.
    std::sort(temp.begin(), temp.end());
    
    // Select the first k elements from the sorted list; if there are fewer
    // than k elements, keep the entire list.
    _size = std::min(total, k);
    _entries.reset(new ViterbiSemiring[_size]);
    for (int i = 0; i < _size; i++)
      _entries[i] = temp[i];

    return (*this);
  }
  
  static KBestViterbiSemiring one(const size_t numFeatures) {
    return KBestViterbiSemiring(LogWeight(1));
  }

  static KBestViterbiSemiring zero(const size_t numFeatures) {
    return KBestViterbiSemiring(LogWeight());
  }
  
  boost::shared_array<ViterbiSemiring> entries() const {
    return _entries;
  }
  
  int size() const {
    return _size;
  }
  
  // Although insideOutside() cannot be called with this semiring, we need to
  // define this type in order to get everything to compile.
  typedef char InsideOutsideResult;
  
  // This variable is the "k" in k-best. By using a static variable, we can
  // use the existing Inference::inside() function without modification for
  // determining the k-best paths. Otherwise, we would have to create a way to
  // pass the "k" around. This variable initialized in latent_struct.cpp.
  static int k;
  
private:
  
  int _size; // the number of (score,bp) entries in this vector
  
  boost::shared_array<ViterbiSemiring> _entries;
};

#endif
