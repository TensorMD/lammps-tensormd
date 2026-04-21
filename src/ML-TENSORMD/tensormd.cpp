/* ----------------------------------------------------------------------
  MIT License
  Copyright (c) 2026
  [Chen Xin (chen_xin@iapcm.ac.cn), Ouyang Yucheng (ouyangyucheng@live.com)]
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ----------------------------------------------------------------------- */

#include "tensormd.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/kernels_utils.h"
#include "kernels/kernels_fusion.h"
#include "tensormd_strategy.h"
#include "tensormd_v1.h"
#include "utils/tensor_debug.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
TensorMD<Scalar>::TensorMD(Device dev, int mpi_rank_id)
    : timer(dev, sizeof(Scalar)),
      dynamic_pool(dev, "TensorMD::Pool::dynamic", true) {
  context = new TensorMDContext<Scalar>(dev, mpi_rank_id);
  context->cutforce = 0.0;
  context->cutforcesq = 0.0;

  // Call setup device at the very first to avoid device mismatch bug!
  context->verbose_device_map = TENSORMD_VERBOSE_DEVICE_MAP;
  Kernels::setup_device(this->context);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
TensorMD<Scalar>::~TensorMD() {
  Kernels::finalize(this->context);
  delete encoder;
  delete energy_head;
  delete context;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_global(const int* fmap, int atom_ntypes) {
  // Check if TensorData is initialized properly.
  if (context == nullptr || context->data == nullptr) {
    throw std::runtime_error(
        "[TensorMD]: TensorMDContext is not initialized properly!");
  }

  TensorDims dims = context->data->dims;
  if (dims.m <= 0 || dims.m > 6 || dims.k < 1 || dims.b < 1) {
    throw std::runtime_error(
        "[TensorMD]: TensorData is not initialized properly!");
  }

  // Check if cutforce is set properly.
  if (this->context->cutforce == 0.0) {
    throw std::runtime_error("[TensorMD]: cutforce is not set properly!");
  } else {
    this->context->cutforcesq = std::pow(this->context->cutforce, 2);
  }

  // Fill the fixed-size constant tensors (T and eltij_pos)
  std::vector<int> eltij_pos(context->data->ntypes * context->data->ntypes);
  fill_fixed_size_tensors(eltij_pos.data());

  // Alloc fixed size tensors
  this->context->data->allocate_fixed_size_tensors();
  this->context->lmpdata.alloc_fixed_size_arrays(atom_ntypes);

  // Call the initialize kernel: copy eltij_pos and numbers (if not nullptr)
  // to device
  Kernels::initialize(const_cast<const int*>(eltij_pos.data()), fmap, numbers,
                      this->context);

  // Update context pointers
  this->context->encoder = &this->encoder->get_context();
  this->context->head = &this->energy_head->get_context();
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_interpolation(
    std::vector<std::string>& lmp_species, Scalar rmin, Scalar delta) {
  if (this->context->cutforce == 0.0) {
    throw std::runtime_error("[TensorMDCore]: Invalid rcut for interpolation.");
  }
  this->encoder->setup_interpolation(lmp_species, rmin, delta);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_local(int nlocal, int cmax, const int* typenums,
                                   bool require_vatom) {
  // First we may need to resize the TensorData to match the current
  // nlocal and cmax.
  this->context->data->resize(nlocal, cmax, require_vatom);

  // Then we setup the timer if it is not initialized.
  if (!timer.is_initialized()) {
    const int b = this->context->data->dims.b;
    const int d = this->context->data->dims.d;
    const int k = this->context->data->dims.k;
    const int m = this->context->data->dims.m;

    // Setup the timer and set FLOPs factors for strategy-irrelevant kernels.
    timer.setup(b, d, k, m, this->encoder->is_interpolatable(),
                this->encoder->get_interp_dr(), require_vatom);
  }

  // Call the timer step function
  timer.step(nlocal, cmax, this->context->lmpdata.imax);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::sync_lammps_data(int* ilist, int inum, int nlocal,
                                        int nghost, int imax, int* type,
                                        int* fmap, double** x, int* numneigh,
                                        int** firstneigh, double** f,
                                        double** vatom) {
  // Setup the page parameters
  this->context->lmpdata.resize(inum, imax, nlocal, nghost, vatom != nullptr);
  Kernels::sync_lammps_data(ilist, type, fmap, x, numneigh, firstneigh, f,
                            vatom, this->context->lmpdata);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::build_neighbor_list(int ago, double cutneigh,
                                           double sublo[3], double subhi[3]) {
  this->context->lmpdata.set_domain_and_neighbor(ago, cutneigh, sublo, subhi);
  if (ago == 0) {
    Kernels::build_neighbor_list(this->context->lmpdata);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int TensorMD<Scalar>::get_exact_cmax() {
  return Kernels::get_exact_cmax(this->context);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::compute(int* typenums, double* etemp,
                               bool use_atom_etemp,
                               FreeEnergyResult<double>& result, double** f,
                               double** vatom, double scale) {
  // Setup Tensors
  Kernels::setup_tensors(this->context);

  if (STRATEGY == Strategy::Scale) {
    // The scale mode uses one hero kernel to handle all calculations.
    const int n = this->context->data->dims.a;
    if (n == 0) return;
    dynamic_pool.grow(n);
    dynamic_pool.reset();
    d_eatoms = dynamic_pool.request(n);
    Kernels::Fusion::calc_all_fused(context, result.E.toten, d_eatoms);
    if (result.E.calc_atom() && n > 0) {
      // Maybe we should implement a copy_to_host function in the memory pool?
      h_eatoms.resize(n);
      Kernels::Utils::access_data_on_host(d_eatoms, n, this->context->device,
                                          h_eatoms);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int i = 0; i < n; ++i) {
        result.E.eatom[i] = static_cast<double>(h_eatoms[i]) * scale;
      }
    }
  } else {
    if (STRATEGY == Strategy::Exact || STRATEGY == Strategy::Baseline) {
      // The normal
      this->encoder->forward(this->context->data);
      Kernels::calc_P(this->context->data);
      Kernels::calc_G(this->context->data);
    } else if (STRATEGY == Strategy::Speed) {
      // The production mode will write P_kdba back to device memory
      Kernels::Fusion::calc_descriptors_fused<Scalar, true>(this->context);
    } else if (STRATEGY == Strategy::Production) {
      // The production mode will not write P_kdba back to device memory
      // This can save some memory, but it brings redundant calculations in the
      // backward fusion stage.
      Kernels::Fusion::calc_descriptors_fused<Scalar, false>(this->context);
    }

    // Calculate total energy
    if (energy_head->is_finite_temp_head()) {
      compute_free_energy(typenums, etemp, use_atom_etemp, result, scale);
    } else {
      compute_energy(typenums, result.E, scale);
    }

    if (STRATEGY == Strategy::Exact || STRATEGY == Strategy::Baseline) {
      // Start backward propagation
      Kernels::calc_W(this->context->data);
      // Calculate Fa (angular forces) first for memory reuse.
      Kernels::calc_angular_forces(this->context->data);
      // Calculate Fr (radial forces)
      this->encoder->backward(this->context->data);
    } else if (STRATEGY == Strategy::Speed) {
      // The speed mode will read P_kbda from device memory
      Kernels::Fusion::calc_forces_fused<Scalar, true>(this->context);
    } else if (STRATEGY == Strategy::Production) {
      // The production model will recalculate P_kdba on-the-fly
      Kernels::Fusion::calc_forces_fused<Scalar, false>(this->context);
    }
  }

#if defined(TENSORMD_DEBUG)
  may_dump_tensors();
#endif

  // Scatter sum Fr and Fa to get final forces
  Kernels::sum_forces(f, vatom, scale, this->context);

  // Accumulate the recorded times every step
  timer.accumulate();
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double TensorMD<Scalar>::get_total_memory_usage() const {
  double usage = this->context->data->get_memory_usage();
  usage += this->context->lmpdata.get_memory_usage();
  usage += this->encoder->get_memory_usage();
  usage += this->energy_head->get_memory_usage();
  usage += Kernels::get_neighbor_list_memory_usage(this->context->device);
  usage += dynamic_pool.get_memory_usage();
  return usage;
}

template <typename Scalar>
std::map<std::string, double> TensorMD<Scalar>::get_memory_usages() const {
  std::map<std::string, double> usages;
  usages["Device::TensorData"] = this->context->data->get_memory_usage();
  usages["Device::LAMMPSData"] = this->context->lmpdata.get_memory_usage();
  usages["Device::Encoder"] = this->encoder->get_memory_usage();
  usages["Device::EnergyHead"] =
      this->energy_head->get_memory_usage() + dynamic_pool.get_memory_usage();
  double neighbor_usage =
      Kernels::get_neighbor_list_memory_usage(this->context->device);
  usages["Device::Neighbor"] = neighbor_usage;
  return usages;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::may_dump_tensors() {
  const char* var = std::getenv("TENSORMD_DUMP_TENSORS");
  if (var == nullptr) {
    return;  // No environment variable set, skip dumping
  }
  std::string tensor_list(var);
  std::vector<std::string> tensors_to_dump;
  if (!tensor_list.empty()) {
    size_t pos = 0;
    while ((pos = tensor_list.find(',')) != std::string::npos) {
      tensors_to_dump.push_back(tensor_list.substr(0, pos));
      tensor_list.erase(0, pos + 1);
    }
    if (!tensor_list.empty()) {
      tensors_to_dump.push_back(tensor_list);
    }
  }
  TensorData<Scalar>* data = this->context->data;
  const int a = data->dims.a;
  const int b = data->dims.b;
  const int c = data->dims.c;
  const int d = data->dims.d;
  const int k = data->dims.k;
  const int m = data->dims.m;
  Device device = data->device;
  int mpiid = context->mpiid;
  for (const auto& tensor_name : tensors_to_dump) {
    if (tensor_name == "R") {
      TensorMap<Tensor<Scalar, 3>> R_t(data->R, c, b, a);
      Debug::dump_tensor(device, R_t, "R", mpiid);
    } else if (tensor_name == "drdrx") {
      TensorMap<Tensor<Scalar, 4>> drdrx_t(data->drdrx, 3, c, b, a);
      Debug::dump_tensor(device, drdrx_t, "drdrx", mpiid);
    } else if (tensor_name == "M") {
      TensorMap<Tensor<Scalar, 4>> M_t(data->M, d, c, b, a);
      Debug::dump_tensor(device, M_t, "M", mpiid);
    } else if (tensor_name == "H") {
      TensorMap<Tensor<Scalar, 4>> H_t(data->H, k, c, b, a);
      Debug::dump_tensor(device, H_t, "H", mpiid);
    } else if (tensor_name == "dHdr") {
      TensorMap<Tensor<Scalar, 4>> dHdr_t(data->dHdr, k, c, b, a);
      Debug::dump_tensor(device, dHdr_t, "dHdr", mpiid);
    } else if (tensor_name == "sij") {
      TensorMap<Tensor<Scalar, 3>> sij_t(data->sij, c, b, a);
      Debug::dump_tensor(device, sij_t, "sij", mpiid);
    } else if (tensor_name == "dsij") {
      TensorMap<Tensor<Scalar, 3>> dsij_t(data->dsij, c, b, a);
      Debug::dump_tensor(device, dsij_t, "dsij", mpiid);
    } else if (tensor_name == "P") {
      if (TENSORMD_PGW_K_CONTINUOUS) {
        TensorMap<Tensor<Scalar, 4>> P_t(data->P, k, d, b, a);
        Debug::dump_tensor(device, P_t, "P", mpiid);
      } else {
        TensorMap<Tensor<Scalar, 4>> P_t(data->P, d, k, b, a);
        Debug::dump_tensor(device, P_t, "P", mpiid);
      }
    } else if (tensor_name == "G") {
      if (TENSORMD_PGW_K_CONTINUOUS) {
        TensorMap<Tensor<Scalar, 4>> G_t(data->G, k, m, b, a);
        Debug::dump_tensor(device, G_t, "G", mpiid);
      } else {
        TensorMap<Tensor<Scalar, 4>> G_t(data->G, m, k, b, a);
        Debug::dump_tensor(device, G_t, "G", mpiid);
      }
    } else if (tensor_name == "dEdG") {
      if (TENSORMD_PGW_K_CONTINUOUS) {
        TensorMap<Tensor<Scalar, 4>> dEdG_t(data->dEdG, k, m, b, a);
        Debug::dump_tensor(device, dEdG_t, "dEdG", mpiid);
      } else {
        TensorMap<Tensor<Scalar, 4>> dEdG_t(data->dEdG, m, k, b, a);
        Debug::dump_tensor(device, dEdG_t, "dEdG", mpiid);
      }
    } else if (tensor_name == "W") {
      if (TENSORMD_PGW_K_CONTINUOUS) {
        TensorMap<Tensor<Scalar, 4>> W_t(data->W, k, d, b, a);
        Debug::dump_tensor(device, W_t, "W", mpiid);
      } else {
        TensorMap<Tensor<Scalar, 4>> W_t(data->W, d, k, b, a);
        Debug::dump_tensor(device, W_t, "W", mpiid);
      }
    } else if (tensor_name == "U") {
      TensorMap<Tensor<Scalar, 4>> U_t(data->U, k, c, b, a);
      Debug::dump_tensor(device, U_t, "U", mpiid);
    } else if (tensor_name == "V") {
      TensorMap<Tensor<Scalar, 4>> V_t(data->V, d, c, b, a);
      Debug::dump_tensor(device, V_t, "V", mpiid);
    } else if (tensor_name == "Fr") {
      TensorMap<Tensor<Scalar, 4>> Fr_t(data->Fr, 3, c, b, a);
      Debug::dump_tensor(device, Fr_t, "Fr", mpiid);
    } else if (tensor_name == "Fa") {
      TensorMap<Tensor<Scalar, 4>> Fa_t(data->Fa, 3, c, b, a);
      Debug::dump_tensor(device, Fa_t, "Fa", mpiid);
    } else if (tensor_name == "ijlist") {
      TensorMap<Tensor<int, 4>> ijlist_t(data->ijlist, 2, c, b, a);
      Debug::dump_tensor<int>(device, ijlist_t, "ijlist", mpiid);
    } else if (tensor_name == "tlist") {
      TensorMap<Tensor<int, 3>> tlist_t(data->tlist, 1, 1, a);
      Debug::dump_tensor<int>(device, tlist_t, "tlist", mpiid);
    } else if (tensor_name == "zijlist") {
      TensorMap<Tensor<int, 4>> zijlist_t(data->zijlist, 2, c, b, a);
      Debug::dump_tensor<int>(device, zijlist_t, "zijlist", mpiid);
    } else if (tensor_name == "tijlist") {
      TensorMap<Tensor<int, 3>> tijlist_t(data->tijlist, c, b, a);
      Debug::dump_tensor<int>(device, tijlist_t, "tijlist", mpiid);
    } else {
      if (mpiid == 0) {
        std::cout << "[TensorMD] Skip dumping tensor " << tensor_name
                  << std::endl;
      }
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class TensorMD<float>;
template class TensorMD<double>;

}  // namespace TENSORMD