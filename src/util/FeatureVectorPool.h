/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _POOL_H
#define _POOL_H

template <typename T> class FeatureVector;

#include "FeatureVector.h"
#include <assert.h>
#include <iostream>
#include <list>
#include <vector>
using namespace std;
#include <tr1/unordered_map>
using tr1::unordered_map;

template <typename T>
class FeatureVectorPool {

  public:
  
    FeatureVectorPool(int dim, int size, bool expandable);
    
    ~FeatureVectorPool();
    
    FeatureVector<T>* get();
    FeatureVector<T>* get(const list<int>& indices);
    FeatureVector<T>* get(const unordered_map<int,T>& featureCounts);
    
    void release(FeatureVector<T>* fv);

  private:
  
    FeatureVector<T>* fetchNext();
    
    const int _dim;
  
    vector<FeatureVector<T>* > _fvs;
    
    FeatureVector<T>* _firstAvailable;
    
    bool _expandable;
};

template <typename T>
FeatureVectorPool<T>::FeatureVectorPool(int dim, int size, bool expandable) :
  _dim(dim), _expandable(expandable) {
  _fvs.resize(size);
  for (size_t i = 0; i < _fvs.size()-1; i++) {
    _fvs[i] = new FeatureVector<T>(_dim, true);
    _fvs[i]->setNext(_fvs[i+1]);
    _fvs[i]->setPoolOwner(this);
  }
  _fvs[size-1] = new FeatureVector<T>(_dim, true);
  _fvs[size-1]->setNext(0);
  _fvs[size-1]->setPoolOwner(this);
  _firstAvailable = _fvs[0];
}

template <typename T>
FeatureVectorPool<T>::~FeatureVectorPool() {
  cout << "Size of FeatureVectorPool when destructor called: " << _fvs.size()
    << endl;
  FeatureVector<T>* fv = 0;
  typename vector<FeatureVector<T>* >::iterator it;
  for (it = _fvs.begin(); it != _fvs.end(); ++it) {
    fv = *it;
    assert(fv);
    assert(fv->isOwnedByPool());
    delete fv;
  }
}

template <typename T>
void FeatureVectorPool<T>::release(FeatureVector<T>* fv) {
  assert(fv->isOwnedByPool());
  fv->setNext(_firstAvailable);
  _firstAvailable = fv;
}

template <typename T>
FeatureVector<T>* FeatureVectorPool<T>::get() {
  FeatureVector<T>* fv = fetchNext();
  if (fv && fv->reinit()) {
    assert(fv->isOwnedByPool());
    return fv;
  }
  if (fv)
    release(fv);
  fv = new FeatureVector<T>();
  assert(!fv->isOwnedByPool());
  return fv;
}

template <typename T>
FeatureVector<T>* FeatureVectorPool<T>::get(const list<int>& indices) {
  FeatureVector<T>* fv = fetchNext();
  if (fv && fv->reinit(indices)) {
    assert(fv->isOwnedByPool());
    return fv;
  }
  if (fv)
    release(fv);
  fv = new FeatureVector<T>(indices);
  assert(!fv->isOwnedByPool());
  return fv;
}

template <typename T>
FeatureVector<T>* FeatureVectorPool<T>::get(const unordered_map<int,T>&
    featureCounts) {
  FeatureVector<T>* fv = fetchNext();
  if (fv && fv->reinit(featureCounts)) {
    assert(fv->isOwnedByPool());
    return fv;
  }
  if (fv)
    release(fv);
  fv = new FeatureVector<T>(featureCounts);
  assert(!fv->isOwnedByPool());
  return fv;
}

template <typename T>
inline FeatureVector<T>* FeatureVectorPool<T>::fetchNext() {
  FeatureVector<T>* fv = 0;
  if (_firstAvailable == 0) {
    if (!_expandable)
      return 0;
    _fvs.push_back(new FeatureVector<T>(_dim, true));
    fv = _fvs.back();
    fv->setPoolOwner(this);
    fv->setNext(0);
    // _firstAvailable remains set to 0
  }
  else {
    assert(_firstAvailable->isOwnedByPool());
    fv = _firstAvailable;
    _firstAvailable = fv->getNext();
  }
  assert(fv->isOwnedByPool());
  return fv;
}

#endif
