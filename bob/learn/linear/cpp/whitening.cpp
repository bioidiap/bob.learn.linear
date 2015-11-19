/**
 * @date Tue Apr 2 21:08:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/make_shared.hpp>
#include <bob.math/inv.h>
#include <bob.math/lu.h>
#include <bob.math/stats.h>

#include <bob.learn.linear/whitening.h>

namespace bob { namespace learn { namespace linear {

  WhiteningTrainer::WhiteningTrainer()
  {
  }

  WhiteningTrainer::WhiteningTrainer(const WhiteningTrainer& other)
  {
  }

  WhiteningTrainer::~WhiteningTrainer() {}

  WhiteningTrainer& WhiteningTrainer::operator= (const WhiteningTrainer& other)
  {
    return *this;
  }

  bool WhiteningTrainer::operator== (const WhiteningTrainer& other) const
  {
    return true;
  }

  bool WhiteningTrainer::operator!= (const WhiteningTrainer& other) const
  {
    return false;
  }

  void WhiteningTrainer::train(Machine& machine, const blitz::Array<double,2>& ar) const {
    // training data dimensions
    const size_t n_samples = ar.extent(0);
    const size_t n_features = ar.extent(1);
    // machine dimensions
    const size_t n_inputs = machine.inputSize();
    const size_t n_outputs = machine.outputSize();

    // Checks that the dimensions are matching
    if (n_inputs != n_features) {
      boost::format m("machine input size (%u) does not match the number of columns in input array (%d)");
      m % n_inputs % n_features;
      throw std::runtime_error(m.str());
    }
    if (n_outputs != n_features) {
      boost::format m("machine output size (%u) does not match the number of columns in output array (%d)");
      m % n_outputs % n_features;
      throw std::runtime_error(m.str());
    }

    // 1. Computes the mean vector and the covariance matrix of the training set
    blitz::Array<double,1> mean(n_features);
    blitz::Array<double,2> cov(n_features,n_features);
    bob::math::scatter(ar, cov, mean);
    cov /= (double)(n_samples-1);

    // 2. Computes the inverse of the covariance matrix
    blitz::Array<double,2> icov(n_features,n_features);
    bob::math::inv(cov, icov);

    // 3. Computes the Cholesky decomposition of the inverse covariance matrix
    blitz::Array<double,2> whiten(n_features,n_features);
    bob::math::chol(icov, whiten);

    // 4. Updates the linear machine
    machine.setInputSubtraction(mean);
    machine.setInputDivision(1.);
    machine.setWeights(whiten);
    machine.setBiases(0);
    machine.setActivation(boost::make_shared<bob::learn::activation::IdentityActivation>());
  }

}}}
