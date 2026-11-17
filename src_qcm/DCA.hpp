#ifndef DCA_hpp
#define DCA_hpp

/**
 * @file DCA.hpp
 * @brief Dynamical cluster approximation (DCA): a model instance whose lattice Green function is diagonal in the
 * cluster momenta K and built from the dispersion of each K patch, instead of CPT's.
 */

#include "lattice_model_instance.hpp"

//! model instance in the dynamical cluster approximation (one cluster, one band)
struct DCA_instance : lattice_model_instance {
	vector<vector3D<double>> K; //!< cluster momenta (superdual basis)
	matrix<Complex> U0; //!< transform from sites to cluster momenta, U0(K,i) = exp(-i K.r_i)/sqrt(Lc)
	map<int, vector<int>> channels; //!< system (index within the cluster) -> indices of the K its bath carries; systems not listed: found from the bath

	DCA_instance(shared_ptr<lattice_model> _model, int label);
	using lattice_model_instance::cluster_Green_function; // keeps the per-cluster overload visible
	void set_Gcpt(Green_function_k &M) override;
	void set_Gcpt_cluster(Green_function_k &M) override;
	void inverse_Gcpt(const block_matrix<Complex> &Ginv, Green_function_k &M) override;
	Green_function cluster_Green_function(Complex w, bool sig, bool spin_down) override;
	matrix<complex<double>> cluster_self_energy_remix(size_t c, complex<double> w, bool spin_down) override;
	double Potthoff_functional() override;

	vector<Complex> self_energy_K(Complex w, bool spin_down);
	size_t patch(const vector3D<double> &k, vector3D<double> &kt);
	vector<Complex> phases(const vector3D<double> &kt);
	vector<Complex> epsilon(const matrix<Complex> &t, const vector3D<double> &kt);
};

#endif
