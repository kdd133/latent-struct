/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include <assert.h>
#include <cmath>
#include <ostream>
#include "LogWeight.h"
#include "RealWeight.h"

const double RealWeight::kZero = 0.0;

const double RealWeight::kOne = 1.0;

LogWeight RealWeight::convert() const {
  if (_val == RealWeight::kOne)
    return LogWeight(LogWeight::kOne);
  else if (_val == RealWeight::kZero)
    return LogWeight(LogWeight::kZero);
  else {
    if (_val > 0)
      return LogWeight(log(_val));
    else
      return LogWeight(log(-_val), -1);
  }
}

RealWeight operator-(const RealWeight& v)
{
  return RealWeight(-v._val);
}

ostream& operator<<(ostream& out, const RealWeight& w) {
  return out << "(" << (int)w._sign << ")" << w._val;
}
