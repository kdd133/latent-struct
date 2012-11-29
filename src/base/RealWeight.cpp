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


LogWeight RealWeight::convert() const {
  return LogWeight(_val);
}

RealWeight operator-(const RealWeight& v)
{
  return RealWeight(-v._val);
}

ostream& operator<<(ostream& out, const RealWeight& w) {
  return out << w._val;
}

const RealWeight RealWeight::operator+(const RealWeight& w) const {
  return plus(w);
}

RealWeight& RealWeight::operator+=(const RealWeight& w) {
  plusEquals(w);
  return (*this);
}

RealWeight& RealWeight::operator*=(const RealWeight& w) {
  timesEquals(w);
  return (*this);
}

const RealWeight RealWeight::operator*(const RealWeight& w) const {
  return times(w);
}
