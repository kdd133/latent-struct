/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "BmrmOptimizer.h"
#include "Model.h"
#include "Optimizer.h"
#include "Regularizer.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_deque.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <string>
#include <uQuadProg++.hh>

using namespace boost;
using namespace std;

BmrmOptimizer::BmrmOptimizer(shared_ptr<TrainingObjective> objective,
                             shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _maxIters(250), _quiet(false),
    _noShrinking(false) {
}

int BmrmOptimizer::processOptions(int argc, char** argv) {
  namespace opt = program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
    ("no-shrinking", opt::bool_switch(&_noShrinking),
        "disable the shrinking heuristic")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help"))
    cout << options << endl;
  return 0;
}

Optimizer::status BmrmOptimizer::train(Parameters& w, double& min_Jw,
    double tol) const {
  namespace ublas = numeric::ublas;
  const size_t d = w.getTotalDim();
  assert(d > 0);
  const double beta = _regularizer->getBeta();
  
  const double TINY = 1e-6; // Add this value to D to ensure pos-def.
  const double ALPHA_TOL = 1e-12; // Value below which alpha is considered zero

  // Some variables we'll need to reuse inside the main loop.
  ptr_deque<RealVec> grads;
  RealVec b(1);
  RealMat G(1, 1);
  RealMat copyG(1, 1);
  RealVec wTemp(d);
  RealVec grad_t(d);
  double Remp;
  
  // Since this is (presumably) a convex objective, the starting point should
  // not matter. We'll silently set it to zero here, because taking the
  // starting point from the last EM iteration sometimes causes issues (i.e.,
  // we may be starting at the minimizer).
  w.zero();
  
  // Compute the initial objective value and gradient.
  _objective->valueAndGradient(w, Remp, grad_t);
  min_Jw = (0.5 * beta * w.squaredL2Norm()) + Remp;
  double Jw = 0; // initialized in the loop
  
  if (!_quiet)
    cout << name() << ": Starting objective value is " << min_Jw << endl;
  
  bool converged = false;
  for (size_t t = 1; t <= _maxIters; t++) {
    timer::cpu_timer timer;
    
    // Add the next column to the matrix "A", which we'll actually represent as
    // a ptr_vector of column vectors.
    ublas::vector<double>* g_t = new ublas::vector<double>(d);
    for (size_t i = 0; i < d; i++)
      (*g_t)(i) = grad_t(i);
    grads.push_back(g_t);
    
    const size_t bs = grads.size(); // The current bundle size.

    // Set the quadratic term to G := A'*A.
    // Note: We could rebuild G from scratch each time (and then add tiny
    // values to the diagonal), but that's much slower, e.g.:
    // ublas::matrix<double> G = ublas::prod(ublas::trans(A), A) / _beta;
    G.resize(bs, bs, true); // Note: true --> preserve existing entries
    const size_t j = bs -1;
    for (size_t i = 0; i < bs; i++) {
      const double temp = ublas::inner_prod(grads[i], grads[j]) / beta;
      if (i == j)
        G(i, j) = temp + TINY; // Add a small value to diag to ensure pos-def.
      else {
        G(i, j) = temp;
        G(j, i) = temp;
      }
    }
    
    // Append an entry to b.
    const double b_t = Remp - w.innerProd(grad_t);
    b.resize(bs, true);
    b(bs - 1) = -b_t;
    
    // Encode the constraint: L1-norm(alpha) = 1.
    RealMat CE(bs, 1);
    RealVec ce0(1);    
    for (size_t i = 0; i < bs; i++)
      CE(i, 0) = 1;
    ce0(0) = -1;
    
    // Encode the constraint: alpha >= 0.
    ublas::identity_matrix<double> CI(bs);
    RealVec ci0(bs);
    for (size_t i = 0; i < bs; i++)
      ci0(i) = 0;
    
    // Allocate a vector to hold the solution.
    RealVec alpha(bs);
    
    // Note: The call to solve_quadprog may modify G, so we preserve a copy.
    copyG = G;
    double JwCP;
    try {
      JwCP = uQuadProgPP::solve_quadprog(G, b, CE, ce0, CI, ci0, alpha);
    }
    catch (std::exception& e) {
      cout << "uQuadProgPP threw an exception: " << e.what() << endl;
      return Optimizer::FAILURE;
    }
    if (JwCP == numeric_limits<double>::infinity())
      return Optimizer::FAILURE;
    G = copyG; // Restore the correct version of G.
      
    // Negate the optimal value returned, since BMRM thinks we're maximizing.
    JwCP = -JwCP;
    
    // The qp solution gives us alpha; we need to now compute
    // wTemp = -1/beta*(A*alpha).
    wTemp = alpha(0) * grads[0];
    for (size_t i = 1; i < bs; i++)
      wTemp += alpha(i) * grads[i];
    wTemp /= -beta;
    
    // Set the entries W in our model to wTemp.
    w.setWeights(wTemp.data().begin(), d);
    
    // Recompute the objective value and gradient.
    _objective->valueAndGradient(w, Remp, grad_t);
    Jw = (0.5 * beta * w.squaredL2Norm()) + Remp;
    
    if (Jw < min_Jw)
      min_Jw = Jw;

    double epsilon_t = min_Jw - JwCP;
    if (epsilon_t < 0) {
      cout << name() << ": Warning: epsilon_t = " << epsilon_t << " < 0" <<
          " ... Saying converged.\n";
      epsilon_t = 0;
    }
    
    if (!_quiet) {
      printf("%s: t = %d  bundle = %d  Jw_t = %0.4e  min_Jw = %0.4e  \
JwCP = %0.4e  epsilon_t = %0.4e", name().c_str(), (int)t, (int)bs, Jw,
        min_Jw, JwCP, epsilon_t);
      cout << " timer:" << timer.format();
    }
    
    // Shrinking heuristic: Discard (grad,b) pairs for which alpha == 0.
    // Note: The vector b and matrix G will be appropriately resized the next
    // time through the loop.
    if (!_noShrinking) {
      // Note: We start at bs-2 so as to never discard the most recent cutting
      // plane.
      for (int col = bs - 2; col >= 0; col--) {
        if (alpha(col) <= ALPHA_TOL) {
          // Discard the gradient vector.
          grads.erase(grads.begin() + col);
          // Shift the rightmost entries in b to the left.
          for (int i = col; i < bs - 1; i++)
            b(i) = b(i + 1);
          // Shift the rightmost columns of G to the left.
          for (int j = col; j < bs - 1; j++)
            for (int i = 0; i < bs; i++)
              G(i, j) = G(i, j + 1);
          // Shift the bottom rows of G upward.
          for (int i = col; i < bs - 1; i++)
            for (int j = 0; j < bs; j++)
              G(i, j) = G(i + 1, j);
        }
      }
    }
    
    if (epsilon_t <= tol) {
      if (!_quiet) {
        cout << name() << ": Convergence detected; objective value " << min_Jw
          << endl;
      }
      converged = true;
      break;
    }
  }
  
  if (!converged) {
    cout << name() << ": Max iterations reached; objective value " << Jw
      << endl;
    return Optimizer::MAX_ITERS_CONVEX;
  }  
  return Optimizer::CONVERGED;
}
