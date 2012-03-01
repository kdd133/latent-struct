/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Example.h"
#include <assert.h>
#include <list>
#include <vector>
using namespace std;

Dataset::Dataset(size_t partitions) : _numPartitions(partitions) {
  for (size_t i = 0; i < _numPartitions; i++)
    _partitions.push_back(list<Example>());
}

void Dataset::addExample(const Example& ex) {
  _examples.push_back(ex);
  _labels.insert(ex.y());
  _partitions[_examples.size() % _numPartitions].push_back(ex);
}

void Dataset::clear() {
  _examples.clear();
  _labels.clear();
  for (size_t i = 0; i < _numPartitions; i++)
    _partitions[i].clear();
}
