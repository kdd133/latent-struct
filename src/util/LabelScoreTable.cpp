/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "LabelScoreTable.h"
#include <assert.h>
#include <boost/multi_array.hpp>
#include <boost/thread/mutex.hpp>

using boost::extents;
using boost::mutex;
using std::size_t;

LabelScoreTable::LabelScoreTable(size_t t, size_t k) {
  _scores.resize(extents[t][k]);
  for (size_t i = 0; i < t; i++)
    for (size_t j = 0; j < k; j++)
      _scores[i][j] = 0.0;
}

void LabelScoreTable::setScore(size_t i, size_t y, double score) {
  assert(i < _scores.shape()[0]);
  assert(y < _scores.shape()[1]);
  mutex::scoped_lock lock(_flag);
  _scores[i][y] = score;
}

double LabelScoreTable::getScore(size_t i, size_t y) {
  assert(i < _scores.shape()[0]);
  assert(y < _scores.shape()[1]);
  mutex::scoped_lock lock(_flag);
  return _scores[i][y];
}
