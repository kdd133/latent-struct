/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGWEIGHT_H
#define _LOGWEIGHT_H

#include <fst/float-weight.h>
#include <ostream>
using std::ostream;

class RealWeight;

class LogWeight {

  public:
    LogWeight(double value = kZero, short s = 1) : _val(value), _sign(s) {}
    
    LogWeight(fst::LogWeightTpl<double> w) : _val(-w.Value()), _sign(1) {}
    
    inline double value() const { return _val; }
    
    inline int sign() const { return (int)_sign; }

    LogWeight plus(const LogWeight d) const;
    
    LogWeight times(const LogWeight d) const;

    void plusEquals(const LogWeight d);
    
    void timesEquals(const LogWeight d);
    
    RealWeight convert() const;
    
    operator double() const;
    
    friend LogWeight operator-(const LogWeight& v);
    
    friend ostream& operator<<(ostream& out, const LogWeight& w);
    
    
    static const double kZero;
    
    static const double kOne;
    
  private:
    double _val;
    
    char _sign; // set to 1 if the corresponding real value is positive; or -1
    
};

#endif
