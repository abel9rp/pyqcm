/**
 * @file DCA.cpp
 * @brief Dynamical cluster approximation: DCA_instance (see DCA.hpp).
 *
 * The lattice Green function at k = K + kt (kt in the patch of K) is
 *   G = U(kt)^dagger diag_K[(w - epsilon(K+kt) - Sigma_K)^-1] U(kt),   U(kt)_{Ki} = U0_{Ki} exp(-i kt.r_i),
 * with epsilon(K+kt) = [U(kt) t(kt) U(kt)^dagger]_KK. In this (periodic) gauge, averages and periodizations work as
 * with CPT; the average over k of U0^dagger diag_K(...) U0 (set_Gcpt_cluster) is the coarse-grained Green function
 * that the CDMFT host is built from. Sigma_K comes from the system(s) whose bath carries K (self_energy_K).
 */

#include <algorithm>
#include "DCA.hpp"
#include "QCM.hpp"

//! true if two wavevectors (superdual basis) differ by a reciprocal lattice vector
static bool same_momentum(lattice_model &m, const vector3D<double> &k1, const vector3D<double> &k2)
{
	vector3D<double> d = m.physdual.to(m.superdual.from(k1 - k2));
	return fabs(d.x-round(d.x)) < 1e-6 and fabs(d.y-round(d.y)) < 1e-6 and fabs(d.z-round(d.z)) < 1e-6;
}



//==============================================================================
/**
 Builds the cluster momenta K, the Lc reciprocal superlattice vectors (integer in the superdual basis) that are
 distinct modulo the reciprocal lattice, and U0(K,i) = exp(-i K.r_i)/sqrt(Lc), with r_i the positions used by
 periodize(). The model is closed by the base constructor.
 */
DCA_instance::DCA_instance(shared_ptr<lattice_model> _model, int _label) : lattice_model_instance(_model, _label)
{
	auto& m = *model;
	if(m.clusters.size() != 1) qcm_throw("DCA: only one cluster per repeated unit is supported");
	if(m.n_band != 1) qcm_throw("DCA: only one-band models are supported");
	if(m.mixing != HS_mixing::normal and m.mixing != HS_mixing::up_down) qcm_throw("DCA: only normal or up_down mixing is supported");

	// representatives of the Lc classes lie in the box [0,Lc)^d of integer superdual components
	int64_t n2 = (m.spatial_dimension > 1) ? m.Lc : 1;
	int64_t n3 = (m.spatial_dimension > 2) ? m.Lc : 1;
	for(int64_t i=0; i<(int64_t)m.Lc; i++)
		for(int64_t j=0; j<n2; j++)
			for(int64_t l=0; l<n3; l++){
				vector3D<double> k(i, j, l);
				if(none_of(K.begin(), K.end(), [&](auto& q){return same_momentum(m, k, q);})) K.push_back(k);
			}
	if(K.size() != m.Lc) qcm_throw("DCA: found "+to_string(K.size())+" cluster momenta for "+to_string(m.Lc)+" sites");

	U0.set_size(m.Lc, m.Lc);
	for(size_t a=0; a<m.Lc; a++)
		for(size_t i=0; i<m.Lc; i++){
			double phi = -K[a]*m.Green_to_position[i]*2*M_PI;
			U0(a,i) = Complex(cos(phi), sin(phi))/sqrt(1.0*m.Lc);
		}
}



//! the patch of k: k = K + kt, kt folded into the cell [-1/2, 1/2) of the superdual basis; returns the index of K
size_t DCA_instance::patch(const vector3D<double> &k, vector3D<double> &kt)
{
	vector3D<double> Kk(floor(k.x+0.5), floor(k.y+0.5), floor(k.z+0.5));
	kt = k - Kk;
	for(size_t a=0; a<K.size(); a++) if(same_momentum(*model, Kk, K[a])) return a;
	qcm_throw("DCA: no cluster momentum matches the wavevector");
	return 0;
}

//! phases exp(-i kt.r_i) that take U0 to U(kt)
vector<Complex> DCA_instance::phases(const vector3D<double> &kt)
{
	vector<Complex> p(U0.r);
	for(size_t i=0; i<p.size(); i++){
		double phi = -(kt*model->Green_to_position[i])*2*M_PI;
		p[i] = Complex(cos(phi), sin(phi));
	}
	return p;
}

//! dispersion of each patch, epsilon(K+kt) = [U(kt) t(kt) U(kt)^dagger]_KK, in the order of K
vector<Complex> DCA_instance::epsilon(const matrix<Complex> &t, const vector3D<double> &kt)
{
	size_t n = U0.r;
	auto p = phases(kt);
	vector<Complex> e(n, 0.0);
	for(size_t a=0; a<n; a++)
		for(size_t i=0; i<n; i++)
			for(size_t j=0; j<n; j++) e[a] += U0(a,i)*p[i]*t(i,j)*conj(U0(a,j)*p[j]);
	return e;
}



//==============================================================================
/**
 Self-energy of each cluster momentum, Sigma_K = [U0 Sigma_s U0^dagger]_KK from the systems s that carry K
 (averaged over them): those set in `channels`; otherwise, with one system, all of them, and with several
 (sub-baths), the K its bath couples to ([U0 Gamma_s U0^dagger]_KK != 0 at large frequency).
 Warns once if a U0 Sigma_s U0^dagger is not diagonal: the bath of s does not couple to definite K points.
 */
vector<Complex> DCA_instance::self_energy_K(Complex w, bool spin_down)
{
	if(!gf_solved) Green_function_solve();
	size_t nK = U0.r;
	auto& clus = model->clusters[0];
	vector<Complex> S(nK, 0.0);
	vector<double> n_carriers(nK, 0.0);
	auto rotate = [&](const matrix<Complex>& A){ // U0 A U0^dagger
		matrix<Complex> B(nK);
		for(size_t a=0; a<nK; a++) for(size_t b=0; b<nK; b++)
			for(size_t i=0; i<nK; i++) for(size_t j=0; j<nK; j++) B(a,b) += U0(a,i)*A(i,j)*conj(U0(b,j));
		return B;
	};
	for(int s=0; s<clus.nsys; s++){
		size_t sys = model->nsys*label + s + clus.sys_start;
		auto SK = rotate(ED::self_energy(w, spin_down, sys));
		vector<bool> carried(nK, true);
		if(auto it = channels.find(s); it != channels.end()){
			carried.assign(nK, false);
			for(int a : it->second){
				if(a < 0 or a >= (int)nK) qcm_throw("DCA: cluster momentum index "+to_string(a)+" out of range for system "+to_string(s));
				carried[a] = true;
			}
		}
		else if(clus.nsys > 1){
			auto GK = rotate(ED::hybridization_function(Complex(0.0, 100.0), spin_down, sys));
			double gmax = 0.0;
			for(size_t a=0; a<nK; a++) gmax = max(gmax, abs(GK(a,a)));
			for(size_t a=0; a<nK; a++) carried[a] = abs(GK(a,a)) > 1e-6*gmax;
		}
		double diag = 0.0, off = 0.0;
		for(size_t a=0; a<nK; a++) for(size_t b=0; b<nK; b++) (a == b ? diag : off) = max(a == b ? diag : off, abs(SK(a,b)));
		static bool warned = false; // once per run: this is called at every frequency
		if(off > 1e-6*max(diag, 1.0) and !warned){ // floor: at U = 0, Sigma is rounding noise
			warned = true;
			cout << "DCA WARNING: the self-energy of system " << s << " is not diagonal in the cluster momenta (max off-diagonal "
			     << off << ", max diagonal " << diag << "): its bath does not couple to definite K points (e.g. a real cos/sin"
			     << " combination of K and -K). Only the diagonal is used." << endl;
		}
		for(size_t a=0; a<nK; a++) if(carried[a]){
			S[a] += SK(a,a);
			n_carriers[a] += 1.0;
		}
	}
	for(size_t a=0; a<nK; a++){
		if(n_carriers[a] == 0.0) qcm_throw("DCA: no system has a bath coupled to cluster momentum #"+to_string(a));
		S[a] /= n_carriers[a];
	}
	return S;
}



//! the cluster Green function, with Sigma_K (filled here, once per frequency: the k-integrals only read it)
Green_function DCA_instance::cluster_Green_function(Complex w, bool sig, bool spin_down)
{
	Green_function G = lattice_model_instance::cluster_Green_function(w, sig, spin_down);
	G.sigma_K = self_energy_K(w, spin_down);
	return G;
}

//! the cluster self-energy seen by the lattice: U0^dagger diag(Sigma_K) U0
matrix<complex<double>> DCA_instance::cluster_self_energy_remix(size_t c, complex<double> w, bool spin_down)
{
	auto S = self_energy_K(w, spin_down);
	matrix<Complex> g(U0.r);
	for(size_t i=0; i<U0.r; i++)
		for(size_t j=0; j<U0.r; j++)
			for(size_t a=0; a<U0.r; a++) g(i,j) += conj(U0(a,i))*S[a]*U0(a,j);
	return g;
}



//==============================================================================
/**
 The DCA lattice Green function at M.k, in the periodic gauge (see the top of this file).
 @param M [in, out]  Green_function_k object
 */
void DCA_instance::set_Gcpt(Green_function_k &M)
{
	if(M.G.sigma_K.size() == 0) qcm_throw("DCA: the cluster Green function was computed without Sigma_K");
	set_V(M);
	vector3D<double> kt;
	patch(M.k, kt);
	auto e = epsilon(M.t, kt);
	size_t n = U0.r;
	vector<Complex> g(n);
	for(size_t a=0; a<n; a++) g[a] = 1.0/(M.G.w - e[a] - M.G.sigma_K[a]);
	auto p = phases(kt);
	to_zero(M.Gcpt.v);
	for(size_t i=0; i<n; i++)
		for(size_t j=0; j<n; j++){
			for(size_t a=0; a<n; a++) M.Gcpt(i,j) += conj(U0(a,i))*g[a]*U0(a,j);
			M.Gcpt(i,j) *= conj(p[i])*p[j];
		}
}

//! the same in the cluster gauge, U0^dagger diag_K(...) U0: its average over k is the coarse-grained Green function
void DCA_instance::set_Gcpt_cluster(Green_function_k &M)
{
	set_Gcpt(M);
	vector3D<double> kt;
	patch(M.k, kt);
	auto p = phases(kt);
	for(size_t i=0; i<p.size(); i++)
		for(size_t j=0; j<p.size(); j++) M.Gcpt(i,j) *= p[i]*conj(p[j]);
}

//! the inverse of the DCA lattice Green function (Ginv is not used: the DCA one is not built from the cluster one)
void DCA_instance::inverse_Gcpt(const block_matrix<Complex> &Ginv, Green_function_k &M)
{
	set_Gcpt(M);
	M.V = M.Gcpt;
	M.V.inverse();
}

double DCA_instance::Potthoff_functional()
{
	qcm_throw("the Potthoff functional is not implemented for DCA");
	return 0.0;
}
