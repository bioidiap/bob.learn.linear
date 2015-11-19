/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Apr 2 21:04:00 2013 +0200
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_LINEAR_WHITENING_H
#define BOB_LEARN_LINEAR_WHITENING_H

#include <bob.learn.linear/machine.h>

namespace bob { namespace learn { namespace linear {

  /**
   * @brief Sets a linear machine to perform a Whitening transform\n
   *
   * Reference:\n
   * "Independent Component Analysis: Algorithms and Applications",
   *   Aapo Hyvärinen, Erkki Oja,
   *   Neural Networks, 2000, vol. 13, p. 411--430\n
   *
   * Given a training set X, this will compute the W matrix such that:\n
   *   \f$W = cholesky(inv(cov(X_{n},X_{n}^{T})))\f$, where \f$X_{n}\f$
   *   corresponds to the center data
   */
  class WhiteningTrainer {

    public: //api

      /**
       * @brief Initializes a new Whitening trainer.
       */
      WhiteningTrainer();

      /**
       * @brief Copy constructor
       */
      WhiteningTrainer(const WhiteningTrainer& other);

      /**
       * @brief Destructor
       */
      virtual ~WhiteningTrainer();

      /**
       * @brief Assignment operator
       */
      WhiteningTrainer& operator=(const WhiteningTrainer& other);

      /**
       * @brief Equal to
       */
      bool operator==(const WhiteningTrainer& other) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const WhiteningTrainer& other) const;

      /**
       * @brief Trains the LinearMachine to perform the Whitening
       */
      virtual void train(Machine& machine, const blitz::Array<double,2>& data) const;
  };

}}}

#endif /* BOB_LEARN_LINEAR_WHITENING_H */
