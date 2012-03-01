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
#include <limits>
#include <ostream>
using namespace std;
#include "LogWeight.h"
#include "RealWeight.h"
#include "Utility.h"

const double LogWeight::kZero = -numeric_limits<double>::infinity();

const double LogWeight::kOne = 0.0;

// See Table 3 in Li & Eisner paper titled:
// "First- and Second-Order Expectation Semirings with Applications..."
LogWeight LogWeight::plus(const LogWeight w) const {
  if (_val == kZero)
    return w;
  else if (w._val == kZero)
    return *this;
  else {
    double la; // natural log of a
    double lb; // natural log of b
    char sa;   // sign bit for a
    char sb;   // sign bit for b
    // Choose a and b s.t. log(a) >= log(b).
    if (_val >= w._val) {
      la = _val;
      lb = w._val;
      sa = _sign;
      sb = w._sign;
    }
    else {
      la = w._val;
      lb = _val;
      sa = w._sign;
      sb = _sign;
    }
    const double x = exp(lb - la);
    if (sa > 0) {
      if (sb > 0)
        return LogWeight(la + Utility::log1Plus(x), 1);
      else // sb < 0
        return LogWeight(la + log(1.0F - x), 1);
    }
    else { // sa < 0
      if (sb > 0)
        return LogWeight(la + log(1.0F - x), -1);
      else { // sb < 0
        return LogWeight(la + Utility::log1Plus(x), -1);
      }
    }
  }
}

void LogWeight::plusEquals(const LogWeight w) {
  const LogWeight result = this->plus(w);
  _val = result._val;
  _sign = result._sign;
}

LogWeight LogWeight::times(const LogWeight w) const {
  if (_val == kZero)
    return *this;
  else if (w._val == kZero)
    return w;
  else
    return LogWeight(_val + w._val, _sign * w._sign);
}

void LogWeight::timesEquals(const LogWeight w) {
  const LogWeight result = this->times(w);
  _val = result._val;
  _sign = result._sign;
}

RealWeight LogWeight::convert() const {
  double v;
  if (_val == LogWeight::kOne)
    v = RealWeight::kOne;
  else if (_val == LogWeight::kZero)
    v = RealWeight::kZero;
  else
    v = exp(_val);
    
  if (_sign)
    return RealWeight(v);
  else
    return RealWeight(-v);
}

LogWeight::operator double() const {
  return _sign * _val;
}

LogWeight operator-(const LogWeight& v)
{
  // The sign refers to the sign in real-space; so, since we're negating in
  // log-space the sign is not flipped. Only the log value is negated.
  return LogWeight(-v._val, v._sign);
}

ostream& operator<<(ostream& out, const LogWeight& w) {
  return out << "(" << (int)w._sign << ")" << w._val;
}
