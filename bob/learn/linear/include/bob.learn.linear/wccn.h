/**
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Apr 9 22:10:00 2013 +0200
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_LINEAR_WCCN_H
#define BOB_LEARN_LINEAR_WCCN_H

#include <bob.learn.linear/machine.h>

namespace bob { namespace learn { namespace linear {

  /**
   * @brief Sets a linear machine to perform a WCCN transform\n
   *
   * Reference 1:\n
   * "Within-class covariance normalization for SVM-based speaker recognition",
   *   Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke,
   *   ICSLP, 2006\n
   * Reference 2:\n
   * "N. Dehah, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet",
   *   Front-end factor analysis for speaker verification,
   *   IEEE TASLP, 2011\n
   *
   * Given a training set X, this will compute the W matrix such that:\n
   *   \f$W = cholesky(inv(cov(X_{n},X_{n}^{T})))\f$, where \f$X_{n}\f$
   *   corresponds to the center data
   */
  class WCCNTrainer {

    public: //api

      /**
       * @brief Initializes a new WCCN trainer.
       */
      WCCNTrainer();

      /**
       * @brief Copy constructor
       */
      WCCNTrainer(const WCCNTrainer& other);

      /**
       * @brief Destructor
       */
      virtual ~WCCNTrainer();

      /**
       * @brief Assignment operator
       */
      WCCNTrainer& operator=(const WCCNTrainer& other);

      /**
       * @brief Equal to
       */
      bool operator==(const WCCNTrainer& other) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const WCCNTrainer& other) const;

      /**
       * @brief Trains the LinearMachine to perform the WCCN
       */
      virtual void train(Machine& machine, const std::vector<blitz::Array<double, 2>>& data) const;

  };

}}}

#endif /* BOB_LEARN_LINEAR_WCCN_H */
