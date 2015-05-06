/**
 * @date Tue Jun  5 16:54:27 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#ifndef BOB_LEARN_LINEAR_BIC_H
#define BOB_LEARN_LINEAR_BIC_H

#include <blitz/array.h>
#include <bob.io.base/HDF5File.h>

namespace bob { namespace learn { namespace linear {
  /**
   * This class computes the Bayesian Intrapersonal/Extrapersonal Classifier (BIC),
   * (see "Beyond Eigenfaces: Probabilistic Matching for Face Recognition" from Moghaddam, Wahid and Pentland)
   * which estimates the posterior probability that the given <b>image difference vector</b>
   * is of the intrapersonal class, i.e., both images stem from the same person.
   *
   * There are two possible implementations of the BIC:
   * <ul>
   * <li> "The Bayesian Intrapersonal/Extrapersonal Classifier" from Teixeira.<br>
   *    A full projection of the data is performed. No prior for the classes has to be selected.</li>
   *
   * <li> "Face Detection and Recognition using Maximum Likelihood Classifiers on Gabor Graphs" from Günther and Würtz.<br>
   *    Only mean and variance of the difference vectors are calculated. There is no subspace truncation and no priors.</li>
   * </ul>
   * In any of the two implementations, the resulting score (using the forward() method) is a log-likelihood estimate,
   * using Mahalanobis(-like) distance measures.
   *
   */
  class BICMachine
  {
    public:
      //! generates an empty BIC machine
      BICMachine(bool use_DFFS = false);

      //! Copy constructor
      BICMachine(const BICMachine& other);

      //! Load from file
      BICMachine(bob::io::base::HDF5File& hdf5);

      //! Assignment Operator
      BICMachine& operator =(const BICMachine &other);

      //! Equality operator
      bool operator ==(const BICMachine& other) const;

      //! Inequality operator
      bool operator !=(const BICMachine& other) const;

      //! Similarity operator
      bool is_similar_to(const BICMachine& other, const double r_epsilon=1e-5, const double a_epsilon=1e-8) const;

      //! computes the BIC probability score for the given input difference vector
      double forward_(const blitz::Array<double,1>& input) const;

      //! performs some checks before calling the forward_ method
      double forward (const blitz::Array<double,1>& input) const;

      //! sets the IEC vectors of the given class (used by the trainer only; not bound to python)
      void setIEC(bool clazz, const blitz::Array<double,1>& mean, const blitz::Array<double,1>& variances, bool copy_data = false);

      //! sets the BIC projection details of the given class (used by the trainer only; not bound to python)
      void setBIC(bool clazz, const blitz::Array<double,1>& mean, const blitz::Array<double,1>& variances, const blitz::Array<double,2>& projection, const double rho, bool copy_data = false);

      //! loads this machine from the given hdf5 file.
      void load(bob::io::base::HDF5File& hdf5);

      //! saves this machine to the given hdf5 file.
      void save(bob::io::base::HDF5File& hdf5) const;

      //! Use the Distance From Feature Space
      void use_DFFS(bool use_DFFS);

      //! Use the Distance From Feature Space
      bool use_DFFS() const {return m_use_DFFS;}

      //! Expected input dimensionality
      int input_size() const {return m_mu_I.extent(0);}

    private:

      //! initializes internal data storages for the given class
      void initialize(bool clazz, int input_length, int projected_length);

      //! project data?
      bool m_project_data;

      //! mean vectors
      blitz::Array<double, 1> m_mu_I, m_mu_E;
      //! variances (eigenvalues)
      blitz::Array<double, 1> m_lambda_I, m_lambda_E;

      ///
      // only required when projection is enabled
      //! add the distance from feature space?
      bool m_use_DFFS;
      //! projection matrices (PCA)
      blitz::Array<double, 2> m_Phi_I, m_Phi_E;
      //! averaged eigenvalues to calculate DFFS
      double m_rho_I, m_rho_E;
      //! temporary storage
      mutable blitz::Array<double, 1> m_diff_I, m_diff_E;
      mutable blitz::Array<double, 1> m_proj_I, m_proj_E;
  };


  class BICTrainer {
    public:
      //! initializes a BICTrainer to train IEC (without subspace estimation)
      BICTrainer() : m_M_I(0), m_M_E(0) {}
      //! initializes a BICTrainer to train BIC (including subspace truncation)
      BICTrainer(int intra_dim, int extra_dim) : m_M_I(intra_dim), m_M_E(extra_dim) {}

      //! trains the intrapersonal and extrapersonal classes of the given BICMachine
      void train(BICMachine& machine, const blitz::Array<double,2>& intra_differences, const blitz::Array<double,2>& extra_differences) const {
        train_single(false, machine, intra_differences);
        train_single(true, machine, extra_differences);
      }

      //! trains the intrapersonal or the extrapersonal class of the given BICMachine
      void train_single(bool clazz, BICMachine& machine, const blitz::Array<double,2>& differences) const;

    private:

      //! dimensions of the intrapersonal and extrapersonal subspace;
      //! zero if training IEC.
      int m_M_I, m_M_E;
  };


} } } // namespaces

#endif // BOB_LEARN_MISC_BICMACHINE_H
