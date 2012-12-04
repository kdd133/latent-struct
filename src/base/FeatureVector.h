/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _FEATUREVECTOR_H
#define _FEATUREVECTOR_H

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "Alphabet.h"
#include "LogWeight.h"
#include "RealWeight.h"
#include <algorithm>
#include <assert.h>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <ostream>
#include <set>
#include <stdexcept>
#include <tr1/unordered_map>
using boost::numeric::ublas::mapped_matrix;
using boost::numeric::ublas::matrix;
using boost::shared_array;
using boost::shared_ptr;
using namespace std;
using tr1::unordered_map;

// Tell the compiler that the friend function is a specialization of the
// template. (see http://goo.gl/zi2yv)
template <typename Weight> class FeatureVector;
template <typename Weight>
ostream& operator<<(ostream& out, const FeatureVector<Weight>& fv);

template <typename Weight>
class FeatureVector {

  public:
  
    FeatureVector(shared_array<int> indices, shared_array<Weight> values,
        int entries, bool copy = true);
    
    // Creates (and sets to zero) a dense, real-valued vector.
    FeatureVector(const int length, bool allocateIndices = false);
    
    // Creates a zero feature vector.
    FeatureVector();
    bool reinit();
    
    FeatureVector(const boost::numeric::ublas::vector<LogWeight>& vec);
    
    // Creates a sparse, binary-valued vector with given indices set to one.
    FeatureVector(const set<int>& indices);
    bool reinit(const set<int>& indices);
    
    // Creates a sparse, real-valued vector based on the given feature counts.
    FeatureVector(const unordered_map<int,Weight>& featureCounts);
    bool reinit(const unordered_map<int,Weight>& featureCounts);
    
    // Copy constructor (performs a deep copy).
    FeatureVector(const FeatureVector& fv);
 
    int getIndexAtLocation(int location) const;
    
    Weight getValueAtLocation(int location) const;
    
    Weight getValueAtIndex(int location) const;
        
    void addTo(shared_array<Weight>& denseValues, int length,
        Weight scale = Weight(1)) const;
    
    void addTo(FeatureVector& fv, Weight scale = Weight(1)) const;
    
    void addTo(unordered_map<int,Weight>& featureCounts,
        Weight scale = Weight(1)) const;
    
    // Interpret dense as real weights.
    void plusEquals(const double* dense, int len, double scale = 1.0);
    
    void plusEquals(const Weight amount);
    
    void timesEquals(const Weight amount);
    
    int getNumEntries() const;
    
    void setNumEntries(int entries);
    
    int getLength() const;
    
    bool isBinary() const;
    
    bool isDense() const; 
    
    void zero();
    
    void updateLength();
    
    // If this is a sparse vector that has more allocated entries than used
    // entries, reallocate to eliminate the wasted space.
    void pack();
    
    friend ostream& operator<< <>(ostream& out, const FeatureVector& fv);
    
    template<typename A, typename B>
    friend FeatureVector<B> fvConvert(const FeatureVector<A>& source,
        shared_array<B> valuesStorage, int valuesLen);

  private:
  
    shared_array<int> _indices;
    
    shared_array<Weight> _values;
    
    int _entries; // the number of entries in the vector (== length if dense)
    
    int _length; // the length of the vector if it was viewed as being dense
    
    // If the FeatureVector is binary, each value is interpreted as being
    // implicitly multiplied by this factor.
    Weight _scaleFactor;
    
    // The number of entries that were initially allocated (i.e., the maximum
    // number of entries this vector can ever assign).
    int _allocatedEntries; 
    
    FeatureVector& operator=(const FeatureVector& fv);
};

template <typename Weight>
FeatureVector<Weight>::FeatureVector(shared_array<int> indices,
    shared_array<Weight> values, int entries, bool copy) :
  _indices(indices), _values(values), _entries(entries), _length(0),
    _scaleFactor(Weight(1)), _allocatedEntries(entries) {
  if (isDense()) { // dense vector
    _length = _entries;
    if (copy) {
      if (values == 0)
        return; // indices and values are both 0 (i.e., zero feature vector)
      _values.reset(new Weight[_entries]);
      for (int i = 0; i < _entries; i++)
        _values[i] = values[i];
    }
  }
  else { // sparse vector
    updateLength();
    if (copy) {
      _indices.reset(new int[_entries]);
      for (int i = 0; i < _entries; i++)
        _indices[i] = indices[i];
    }
    
    if (!isBinary()) { // real-valued vector
      if (copy) {
        _values.reset(new Weight[_entries]);
        for (int i = 0; i < _entries; i++)
          _values[i] = values[i];        
      }
    }
  }
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector(const int length, bool allocateIndices) :
  _indices(0), _values(0), _entries(length), _length(length),
    _scaleFactor(Weight(1)), _allocatedEntries(length) {
  if (length > 0)
    _values.reset(new Weight[length]);
  if (allocateIndices)
    _indices.reset(new int[length]);
  zero();
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector(
    const boost::numeric::ublas::vector<LogWeight>& vec) :
  _indices(0), _allocatedEntries(0) {
  _values.reset(new Weight[vec.size()]);
  for (size_t i = 0; i < vec.size(); ++i)
    _values[i] = vec(i);
  _entries = _length = vec.size();
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector(const set<int>& indicesList) :
    _indices(0), _values(0), _entries(indicesList.size()),
    _allocatedEntries(_entries) {
  if (_entries == 0)
    return; // return zero vector
  _indices.reset(new int[_entries]);
  reinit(indicesList);
}

template <typename Weight>
bool FeatureVector<Weight>::reinit(const set<int>& indicesList) {
  if ((int)indicesList.size() > _allocatedEntries)
    return false;
  reinit(); // set this to a default zero vector
  _entries = indicesList.size();
  if (_entries == 0)
    return true;
  int i = 0;
  set<int>::const_iterator it = indicesList.begin();
  for (; it != indicesList.end(); ++it)
    _indices[i++] = *it;
  updateLength();
  return true;
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector() :
    _indices(0), _values(0), _allocatedEntries(0) {
  reinit();
}

template <typename Weight>
inline bool FeatureVector<Weight>::reinit() {
  _entries = 0;
  _length = 0;
  _scaleFactor = Weight(1);
  return true;
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector(const unordered_map<int,Weight>&
    featureCounts) :
    _indices(0), _values(0), _entries(featureCounts.size()),
    _allocatedEntries(_entries) {
  if (_entries == 0)
    return; // return zero vector
  _indices.reset(new int[_entries]);
  _values.reset(new Weight[_entries]);
  reinit(featureCounts);
}

template <typename Weight>
bool FeatureVector<Weight>::reinit(const unordered_map<int,Weight>&
    featureCounts) {
  if ((int)featureCounts.size() > _allocatedEntries)
    return false;
  reinit(); // set this to a default zero vector
  _entries = featureCounts.size();
  assert(_indices && _values);
  assert(_entries <= _allocatedEntries);
  if (_entries == 0)
    return true;
  int i = 0;
  typename unordered_map<int,Weight>::const_iterator it = featureCounts.begin();
  for (; it != featureCounts.end(); ++it) {
    assert(it->first >= 0);
    _indices[i] = it->first;
    _values[i] = it->second;
    ++i;
  }
  updateLength();
  return true;
}

template <typename Weight>
void FeatureVector<Weight>::pack() {
  assert(!isDense());
  if (_entries == _allocatedEntries)
    return;
  assert(_indices);
  int* indices = new int[_entries];
  for (int i = 0; i < _entries; i++)
    indices[i] = _indices[i];
  _indices.reset(indices);
  if (!isBinary()) {
    Weight* values = new Weight[_entries];
    for (int i = 0; i < _entries; i++)
      values[i] = _values[i];
    _values.reset(values);
  }
  _allocatedEntries = _entries;
}

template <typename Weight>
FeatureVector<Weight>::FeatureVector(const FeatureVector<Weight>& fv) :
  _indices(0), _values(0), _entries(fv._entries), _length(fv._length),
    _scaleFactor(fv._scaleFactor), _allocatedEntries(_entries) {
  if (fv._indices != 0) {
    int* indsTemp = new int[_entries];
    for (int i = 0; i < _entries; i++)
      indsTemp[i] = fv._indices[i];
    _indices.reset(indsTemp);
  }
  if (fv._values != 0) {
    _values.reset(new Weight[_entries]);
    for (int i = 0; i < _entries; i++)
      _values[i] = fv._values[i];
  }
}

template <typename Weight>
void FeatureVector<Weight>::updateLength() {
  int maxIndex = -1;
  for (int i = 0; i < _entries; i++)
    if (_indices[i] > maxIndex)
      maxIndex = _indices[i];
  _length = maxIndex + 1;
  assert(_length >= _entries);
}

template <typename Weight>
int FeatureVector<Weight>::getIndexAtLocation(int location) const {
  assert(!(location < 0 || location >= _entries));
  if (isDense())
    return location; // dense vector; index == location
  return _indices[location];
}

template <typename Weight>
Weight FeatureVector<Weight>::getValueAtLocation(int location) const {
  assert(!(location < 0 || location >= _entries));
  if (isBinary())
    return _scaleFactor;
  return _values[location];
}

template <typename Weight>
Weight FeatureVector<Weight>::getValueAtIndex(int index) const {
  if (isDense())
    return getValueAtLocation(index); // Location == Index in this case
    
  // TODO: If we knew the indices were sorted, a binary search would be faster.
  int location = -1;
  for (int i = 0; i < _entries; i++) {
    if (index == _indices[i]) {
      location = i;
      break;
    }
  }
  
  if (location == -1) {
    // A missing entry in a sparse vector implies the weight is zero for this
    // feature.
    return Weight(0);
  }
  assert(location >= 0 && location < _entries);
  return getValueAtLocation(location);
}

// adapted from Mallet's SparseVector class
template <typename Weight>
void FeatureVector<Weight>::addTo(shared_array<Weight>& denseValues, int len,
    Weight scale) const {
  assert(len >= _length);
  if (isDense()) {
    for (int i = 0; i < _entries; i++)
      denseValues[i] += (_values[i] * scale);
  }
  else if (isBinary()) { // binary-valued, sparse vector
    scale *= _scaleFactor;
    for (int i = 0; i < _entries; i++)
      denseValues[_indices[i]] += scale;
  }
  else {
    for (int i = 0; i < _entries; i++) // real-valued, sparse vector
      denseValues[_indices[i]] += (_values[i] * scale);
  }
}

template <typename Weight>
void FeatureVector<Weight>::addTo(unordered_map<int,Weight>& featureCounts,
    Weight scale) const {
  if (_entries == 0)
    return;
  assert(!isDense()); // pointless to add a dense vector to a sparse vector
  if (isBinary()) { // binary-valued, sparse vector
    scale *= _scaleFactor;
    if (scale == Weight(0))
      return;
    for (int i = 0; i < _entries; i++)
      featureCounts[_indices[i]] += scale;
  }
  else {
    for (int i = 0; i < _entries; i++) { // real-valued, sparse vector
      const Weight increment = _values[i] * scale;
      if (increment != Weight(0))
        featureCounts[_indices[i]] += increment;
    }
  }
}

template <typename Weight>
void FeatureVector<Weight>::addTo(FeatureVector<Weight>& fv,
    Weight scale) const {
  assert(fv.isDense());
  if (fv._length < _length) {
    // This is a hack that was introduced for the sake of RingInfo's collect()
    // methods, which are typically called from inside() or outside().
    // TODO: In general, we may want to have an addTo method that does not
    // require the destination to be a dense FeatureVector.
    assert(isDense());
    
    shared_array<Weight> oldValues;
    if (fv._length > 0)
      oldValues = fv._values;
    fv._values.reset(new Weight[_length]);
    
    if (fv._length > 0) {
      for (int i = 0; i < fv._length; i++)
        fv._values[i] = oldValues[i];
    }
    fv._length = _length;
    fv._entries = _entries;
    fv._allocatedEntries = _allocatedEntries;
  }
  addTo(fv._values, fv._length, scale);
}

template <typename Weight>
void FeatureVector<Weight>::plusEquals(const double* dense, int len,
    double scale) {
  assert(!(isBinary() || !isDense()));
  assert(_length == len);
  for (int i = 0; i < _entries; i++)
    _values[i] += (Weight(dense[i]) * (Weight)scale);
}

template<>
inline void FeatureVector<LogWeight>::plusEquals(const double* dense, int len,
    double scale) {
  throw logic_error("This function interprets the double* as real weights, and \
cannot be used if the underlying FeatureVector uses LogWeight.");
}

template <typename Weight>
void FeatureVector<Weight>::plusEquals(const Weight amount) {
  assert(!isBinary());
  for (int i = 0; i < _entries; i++)
    _values[i] += amount;
}

template <typename Weight>
void FeatureVector<Weight>::timesEquals(const Weight amount) {
  // We scale binary and real-valued vector differently. For binary vectors, we
  // update the scale factor, which is subsequently used in, e.g.,
  // getValueAtIndex(). For real vectors, we explictly multiply the values.
  if (isBinary())
    _scaleFactor *= amount;
  else {
    for (int i = 0; i < _entries; i++)
      _values[i] *= amount;
  }
}

template <typename Weight>
int FeatureVector<Weight>::getLength() const {
  return _length;
}

template <typename Weight>
int FeatureVector<Weight>::getNumEntries() const {
  return _entries;
}

template <typename Weight>
void FeatureVector<Weight>::setNumEntries(int entries) {
  _entries = entries;
}

// Following Mallet's SparseVector class:
// if indices is 0, the vector will be dense
// if values is 0, the vector will be binary-valued

template <typename Weight>
bool FeatureVector<Weight>::isBinary() const {
  return _values == 0;
}

template <typename Weight>
bool FeatureVector<Weight>::isDense() const {
  return _indices == 0;
}

template <typename Weight>
void FeatureVector<Weight>::zero() {
  if (isBinary())
    _scaleFactor = Weight(0);
  else {
    for (int i = 0; i < _entries; i++)
      _values[i] = Weight(0);
  }
}

template <typename Weight>
ostream& operator<<(ostream& out, const FeatureVector<Weight>& fv) {
  if (fv.isDense()) {
    bool allZero = true;
    for (int i = 0; i < fv._entries; i++) {
      if (fv._values[i] != Weight(0)) { // skip zero entries
        out << i << " " << fv._values[i] << endl;
        if (allZero)
          allZero = false;
      }
    }
    if (allZero)
      out << "(zero vector)" << endl;
  }
  else if (fv.isBinary()) {
    for (int i = 0; i < fv._entries; i++)
      out << "(scale=" << fv._scaleFactor << ")" << fv._indices[i] << endl;
  }
  else {
    for (int i = 0; i < fv._entries; i++)
      out << fv._indices[i] << " " << fv._values[i] << endl;
  }
  return out;
}

template<typename A, typename B>
FeatureVector<B> fvConvert(const FeatureVector<A>& source,
    shared_array<B> valuesStorage, int valuesLen) {
  const bool copy = valuesStorage ? false : true;
  assert(!copy || valuesLen > 0);
  FeatureVector<B> fv;
  fv._indices = source._indices;
  fv._entries = source._entries;
  fv._length = source._length;
  if (source.isBinary()) {
    fv._scaleFactor = source._scaleFactor.convert();
    fv._values.reset(); // i.e., set values to 0
  }
  else {
    fv._scaleFactor = B(1);
    if (copy)
      fv._values.reset(new B[fv._entries]);
    else
      fv._values = valuesStorage;
    for (int i = 0; i < source._entries; i++)
      fv._values[i] = source._values[i].convert();
  }
  return fv;
}

#endif
