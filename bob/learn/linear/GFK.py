#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""

Implementing the algorithm Geodesic Flow Kernel to do transfer learning from the modality A to modality B from the paper

Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

A very good explanation can be found here
http://www-scf.usc.edu/~boqinggo/domainadaptation.html#gfk_section

"""

import bob.io.base
import numpy
import scipy.linalg

import logging

logger = logging.getLogger("bob.learn.linear")


def null_space(A, eps=1e-20):
    """
    Computes the left null space of `A`.
    The left null space of A is the orthogonal complement to the column space of A.

    """
    U, S, V = scipy.linalg.svd(A)

    padding = max(0, numpy.shape(A)[1] - numpy.shape(S)[0])
    null_mask = numpy.concatenate(((S <= eps), numpy.ones((padding,), dtype=bool)), axis=0)
    null_s = scipy.compress(null_mask, V, axis=0)
    return scipy.transpose(null_s)


class GFKMachine(object):
    """
    Geodesic flow Kernel (GFK) Machine.

    This is output of the :py:class:`bob.learn.linear.GFKTrainer`
    """

    def __init__(self, hdf5=None):
        """
        Constructor

        **Parameters**
          hdf5: :py:class:`bob.io.base.HDF5File`
            If is not None, will read all the content of the HDF5File
        """
        self.source_machine = None
        self.target_machine = None
        self.G = None

        if isinstance(hdf5, bob.io.base.HDF5File):
            self.load(hdf5)

    def load(self, hdf5):
        """
        Loads the machine from the given HDF5 file

        **Parameters**

          hdf5: :py:class:`bob.io.base.HDF5File`
            An HDF5 file opened for reading

        """

        assert isinstance(hdf5, bob.io.base.HDF5File)

        # read PCA projector
        hdf5.cd("source_machine")
        self.source_machine = bob.learn.linear.Machine(hdf5)
        hdf5.cd("..")
        hdf5.cd("target_machine")
        self.target_machine = bob.learn.linear.Machine(hdf5)
        hdf5.cd("..")
        self.G = hdf5.get("G")

    def save(self, hdf5):
        """
        Saves the machine to the given HDF5 file

        **Parameters**

          hdf5: :py:class:`bob.io.base.HDF5File`
            An HDF5 file opened for writing
        """

        hdf5.create_group("source_machine")
        hdf5.cd("source_machine")
        self.source_machine.save(hdf5)
        hdf5.cd("..")
        hdf5.create_group("target_machine")
        hdf5.cd("target_machine")
        self.target_machine.save(hdf5)
        hdf5.cd("..")
        hdf5.set("G", self.G)

    def shape(self):
        """
        A tuple that represents the shape of the kernel matrix

        **Returns**
         (int, int) <– The size of the weights matrix
        """
        return self.G.shape

    def compute_principal_angles(self):
        r"""
        Compute the principal angles between source (:math:`P_s`) and target (:math:`P_t`) subspaces in a Grassman which is defined as the following:

        :math:`d^{2}(P_s, P_t) = \sum_{i}( \theta_i^{2} )`,

        """
        Ps = self.source_machine.weights
        Pt = self.target_machine.weights

        # S = cos(theta_1, theta_2, ..., theta_n)
        _, S, _ = numpy.linalg.svd(numpy.dot(Ps.T, Pt))
        thetas_squared = numpy.arccos(S) ** 2

        return numpy.sum(thetas_squared)

    def compute_binetcouchy_distance(self):
        """
        Compute the Binet-Couchy distance between source (:math:`P_s`) and target (:math:`P_t`) subspaces in a Grassman which is defined as the following:

        :math:`d(P_s, P_t) = 1 - (det(P_{s}^{T} * P_{t}^{T}))^{2}`
        """

        # Preparing the source
        Ps = self.source_machine.weights
        Rs = null_space(Ps.T)
        Y1 = numpy.hstack((Ps, Rs))

        # Preraring the target
        Pt = self.target_machine.weights
        Rt = null_space(Pt.T)
        Y2 = numpy.hstack((Pt, Rt))

        return 1 - numpy.linalg.det(numpy.dot(Y1.T, Y2)) ** 2

    def __call__(self, source_domain_data, target_domain_data):
        """
        Compute dot product in the infinity space using the trainer Kernel (G)

        **Parameters**
          source_domain_data: :py:func:`numpy.array`
            Data from the source domain

          target_domain_data: :py:func:`numpy.array`
            Data from the target domain
        """
        source_domain_data = (
                             source_domain_data - self.source_machine.input_subtract) / self.source_machine.input_divide
        target_domain_data = (
                             target_domain_data - self.target_machine.input_subtract) / self.target_machine.input_divide

        return numpy.dot(numpy.dot(source_domain_data, self.G), target_domain_data.T)[0]


class GFKTrainer(object):
    r"""
    Trains the Geodesic Flow Kernel (GFK) that models the domain shift from a certain source linear subspace :math:`P_S` to
    a certain target linear subspaces :math:`P_T`.

    GFK models the source domain and the target domain with d-dimensional linear subspaces and embeds them onto a Grassmann manifold.
    Specifically, let denote the basis of the PCA subspaces for each of the two domains, respectively.
    The Grassmann manifold :math:`G(d,D)` is the collection of all d-dimensional subspaces of the feature vector space :math:`\mathbb{R}^D`.

    The geodesic flow :math:`\phi(t)` between :math:`P_S, P_T` on the manifold parameterizes a path connecting the two subspaces.
    In the beginning of the flow, the subspace is similar to that of the source domain and in the end of the flow,
    the subspace is similar to that of the target.
    The original feature :math:`x` is projected into these subspaces and forms a feature vector of infinite dimensions:

    :math:`z^{\infty} = \phi(t)^T x: t \in [0, 1]`.

    Using the new feature representation for learning, will force the classifiers to NOT lean towards either the source
    domain or the target domain, or in other words, will force the classifier to use domain-invariant features.
    The infinite-dimensional feature vector is handled conveniently by their inner product that gives rise to a positive
    semidefinite kernel defined on the original features,

    :math:`G(x_i, x_j) = x_{i}^T \int_0^1 \! \phi(t)\phi(t)^T  \, \mathrm{d}t x_{j} = x_i^T G x_j`.

    The matrix G can be computed efficiently using singular value decomposition. Moreover, computing the kernel does not require any labeled data.

    More details can be found in:

    Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

    A very good intuition can be found in:
    http://www-scf.usc.edu/~boqinggo/domainadaptation.html#gfk_section

    **Constructor Documentation:**

       - **bob.learn.linear.GFKTrainer** (number_of_subspaces, subspace_dim_source, subspace_dim_target, eps)

        **Parameters**

          number_of_subspaces: `int`
            Number of subspaces for the transfer learning. If set to -1, this value will be estimated automatically. For more information check, Section 3.4.

          subspace_dim_source: `float`
            Energy kept in the source linear subspace

          subspace_dim_target: `float`
            Energy kept in the target linear subspace

          eps: `float`
            Floor value

    """

    def __init__(self, number_of_subspaces=-1, subspace_dim_source=0.99, subspace_dim_target=0.99, eps=1e-20):
        """
        Constructor

        **Parameters**

          number_of_subspaces: `int`
            Number of subspaces for the transfer learning

          subspace_dim_source: `float`
            Energy kept in the source linear subspace

          subspace_dim_target: `float`
            Energy kept in the target linear subspace

          eps: `float`
            Floor value
        """
        self.m_number_of_subspaces = number_of_subspaces
        self.m_subspace_dim_source = subspace_dim_source
        self.m_subspace_dim_target = subspace_dim_target
        self.eps = eps


    def get_best_d(self, Ps, Pt, Pst):
        """
        Get the best value for the number of subspaces

        For more details, read section 3.4 of the paper.

        **Parameters**
          Ps: Source subspace

          Pt: Target subspace

          Pst: Source + Target subspace
        """
        def compute_angles(A, B):
            _, S, _ = numpy.linalg.svd(numpy.dot(A.T, B))
            S[numpy.where(numpy.isclose(S ,1, atol=self.eps)==True)[0]] = 1
            return numpy.arccos(S)

        max_d = min(Ps.shape[1], Pt.shape[1], Pst.shape[1] )
        alpha_d = compute_angles(Ps, Pst)
        beta_d = compute_angles(Pt, Pst)

        d = 0.5 * ( numpy.sin(alpha_d) + numpy.sin(beta_d))

        return numpy.argmax(d)

    def train(self, source_data, target_data, norm_inputs=True):
        """
        Trains the GFK (:py:class:`bob.learn.linear.GFKMachine`)

        **Parameters**
          source_data: :py:func:`numpy.array`
            Data from the source domain

          target_data: :py:func:`numpy.array`
            Data from the target domain

        **Returns**

          machine: :py:class:`bob.learn.linear.GFKMachine`

        """

        source_data = source_data.astype("float64")
        target_data = target_data.astype("float64")

        if(self.m_number_of_subspaces == -1):
            source_target = numpy.vstack((source_data, target_data))
            norm_inputs = True
            logger.info("  -> Automatic search for d. We set norm_inputs=True")


        logger.info("  -> Normalizing data per modality")
        if norm_inputs:
            source, mu_source, std_source = self._znorm(source_data)
            target, mu_target, std_target = self._znorm(target_data)
        else:
            mu_source = numpy.zeros(shape=(source_data.shape[1]))
            std_source = numpy.ones(shape=(source_data.shape[1]))
            mu_target = numpy.zeros(shape=(target_data.shape[1]))
            std_target = numpy.ones(shape=(target_data.shape[1]))
            source = source_data
            target = target_data

        logger.info("  -> Computing PCA for the source modality")
        Ps = self._train_pca(source, mu_source, std_source, self.m_subspace_dim_source)
        logger.info("  -> Computing PCA for the target modality")
        Pt = self._train_pca(target, mu_target, std_target, self.m_subspace_dim_target)
        # self.m_machine                = bob.io.base.load("/idiap/user/tpereira/gitlab/workspace_HTFace/GFK.hdf5")

        # If -1, let's compute the optimal value for d
        if(self.m_number_of_subspaces == -1):
            logger.info("  -> Computing the best value for m_number_of_subspaces")

            source_target, mu_source_target, std_source_target = self._znorm(source_target)
            Pst = self._train_pca(source_target, mu_source_target, std_source_target, min(self.m_subspace_dim_target, self.m_subspace_dim_source))

            self.m_number_of_subspaces = self.get_best_d(Pst.weights, Ps.weights, Pt.weights)
            logger.info("  -> Best m_number_of_subspaces is {0}".format(self.m_number_of_subspaces))

        G = self._train_gfk(numpy.hstack((Ps.weights, null_space(Ps.weights.T))),
                            Pt.weights[:, 0:self.m_number_of_subspaces])

        machine = GFKMachine()
        machine.source_machine = Ps
        machine.target_machine = Pt
        machine.G = G

        return machine

    def _train_gfk(self, Ps, Pt):
        """
        """

        N = Ps.shape[1]
        dim = Pt.shape[1]

        # Principal angles between subspaces
        QPt = numpy.dot(Ps.T, Pt)

        # [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
        A = QPt[0:dim, :].copy()
        B = QPt[dim:, :].copy()

        # Equation (2)
        [V1, V2, V, Gam, Sig] = bob.math.gsvd(A, B)
        V2 = -V2

        # Some sanity checks with the GSVD
        I = numpy.eye(V1.shape[1])
        I_check = numpy.dot(Gam.T, Gam) + numpy.dot(Sig.T, Sig)
        assert numpy.sum(abs(I - I_check)) < 1e-10

        theta = numpy.arccos(numpy.diagonal(Gam))

        # Equation (6)
        B1 = numpy.diag(0.5 * (1 + (numpy.sin(2 * theta) / (2. * numpy.maximum
        (theta, self.eps)))))
        B2 = numpy.diag(0.5 * ((numpy.cos(2 * theta) - 1) / (2 * numpy.maximum(
            theta, self.eps))))
        B3 = B2
        B4 = numpy.diag(0.5 * (1 - (numpy.sin(2 * theta) / (2. * numpy.maximum
        (theta, self.eps)))))

        # Equation (9) of the suplementary matetial
        delta1_1 = numpy.hstack((V1, numpy.zeros(shape=(dim, N - dim))))
        delta1_2 = numpy.hstack((numpy.zeros(shape=(N - dim, dim)), V2))
        delta1 = numpy.vstack((delta1_1, delta1_2))

        delta2_1 = numpy.hstack((B1, B2, numpy.zeros(shape=(dim, N - 2 * dim))))
        delta2_2 = numpy.hstack((B3, B4, numpy.zeros(shape=(dim, N - 2 * dim))))
        delta2_3 = numpy.zeros(shape=(N - 2 * dim, N))
        delta2 = numpy.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = numpy.hstack((V1, numpy.zeros(shape=(dim, N - dim))))
        delta3_2 = numpy.hstack((numpy.zeros(shape=(N - dim, dim)), V2))
        delta3 = numpy.vstack((delta3_1, delta3_2)).T

        delta = numpy.dot(numpy.dot(delta1, delta2), delta3)
        G = numpy.dot(numpy.dot(Ps, delta), Ps.T)

        return G

    def _train_pca(self, data, mu_data, std_data, subspace_dim):
        t = bob.learn.linear.PCATrainer()
        machine, variances = t.train(data)

        # For re-shaping, we need to copy...
        variances = variances.copy()

        # compute variance percentage, if desired
        if isinstance(subspace_dim, float):
            cummulated = numpy.cumsum(variances) / numpy.sum(variances)
            for index in range(len(cummulated)):
                if cummulated[index] > subspace_dim:
                    subspace_dim = index
                    break
            subspace_dim = index
        logger.info("    ... Keeping %d PCA dimensions", subspace_dim)

        machine.resize(machine.shape[0], subspace_dim)
        machine.input_subtract = mu_data
        machine.input_divide = std_data

        return machine

    def _znorm(self, data):
        """
        Z-Normaliza
        """

        mu = numpy.average(data, axis=0)
        std = numpy.std(data, axis=0)

        data = (data - mu) / std

        return data, mu, std
