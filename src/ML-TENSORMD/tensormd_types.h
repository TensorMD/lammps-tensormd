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

#ifndef TENSORMD_TYPES_H
#define TENSORMD_TYPES_H

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>
#include <vector>

#include "tensormd_constants.h"
#include "tensormd_device.h"
#include "tensormd_memory.h"
#include "tensormd_strategy.h"

namespace TENSORMD {

using TENSORMD::STRATEGY;

// Basic data structure for potential energy (Total + Per-Atom)
template <typename Scalar>
struct EnergyResult {
  Scalar toten = 0.0;
  Scalar* eatom = nullptr;

  [[nodiscard]] bool calc_atom() const { return eatom != nullptr; }
  void zero() { toten = 0.0; }
};

// Advaced data structure for finite-temperature potential energy
// Electron free energy, electron internal energy and electron entropy are
// included.
template <typename Scalar>
struct FreeEnergyResult {
  EnergyResult<Scalar> F;  // Free Energy
  EnergyResult<Scalar> E;  // Internal Energy
  EnergyResult<Scalar> S;  // Entropy

  [[nodiscard]] bool calc_atom() const {
    return F.calc_atom() && E.calc_atom() && S.calc_atom();
  }

  void zero() {
    F.toten = 0.0;
    E.toten = 0.0;
    S.toten = 0.0;
  }
};

// Tensor Dimensions Container
struct TensorDims {
  int a{0};         // Atoms (local)
  int b{0};         // Element types
  int c{0};         // Neighbors (max)
  int d{0};         // Unique moments
  int k{0};         // Radial basis functions
  int m{0};         // Angular moments
  const int x = 3;  // Cartesian (fixed)
};

template <typename Scalar>
class TensorData {
 public:
  // Dimensions and Device config
  TensorDims dims;
  Device device;

  // The number of atom types for the corresponding potential.
  // For V1, dims.b and ntypes are always the same.
  // For V2, dims.b is always 1 because it uses a mapping scheme.
  int ntypes = 0;
  int* numbers = nullptr;  // V2 specific, atom types to atomic numbers

  // --- Raw Pointers (Public for direct Kernel access) ---

  // 0: V1, legacy (will be discarded soon)
  // 1: V1, compressed (will be discarded soon)
  // 2: V2, accurate
  int T_proj_algo = 1;

  int* eltij_pos = nullptr;
  int* elt_a_offset = nullptr;  // V1 specific (b > 1)

  // Dynamically sized tensors
  int* ijlist = nullptr;
  int* i2a = nullptr;      // V1 specific (b > 1)
  int* a2i = nullptr;      // V1 specific (b > 1)
  int* zijlist = nullptr;  // V2 specific
  int* tijlist = nullptr;  // V2 specific
  int* tlist = nullptr;    // V2 specific

  Scalar* R = nullptr;
  Scalar* drdrx = nullptr;
  Scalar* M = nullptr;
  Scalar* sij = nullptr;
  Scalar* dsij = nullptr;
  Scalar* H = nullptr;
  Scalar* dHdr = nullptr;
  Scalar* P = nullptr;
  Scalar* G = nullptr;
  Scalar* dEdG = nullptr;
  Scalar* W = nullptr;
  Scalar* U = nullptr;
  Scalar* Up = nullptr;
  Scalar* dHdrp = nullptr;
  Scalar* V = nullptr;
  Scalar* Fr = nullptr;
  Scalar* Fa = nullptr;

  // The constructor initializes the device and fixed dimensions, but does
  // not allocate memory for the tensors.
  // The resize() method is responsible for allocating memory based on the
  // actual system size.
  TensorData(Device dev, int ntypes, int b, int k, int m, int T_proj_algo,
             bool V2 = false)
      : device(dev),
        scalar_pool(dev, "TensorData::Pool::scalar", true),
        int_pool(dev, "TensorData::Pool::integer", true),
        float_pool(dev, "TensorData::Pool::float", true),
        static_int_pool(dev, "TensorData::Pool::static_integer", false),
        V2(V2),
        ntypes(ntypes),
        T_proj_algo(T_proj_algo) {
    // Validate input parameters
    if (b <= 0 || k <= 0 || m < 1 || m > 6) {
      throw std::invalid_argument(
          "Invalid tensor dimensions: b,k > 0 and 1 <= m <= 6.");
    }
    if (b > ntypes) {
      throw std::invalid_argument(
          "TensorData constructor error: b cannot be greater than ntypes.");
    }
    if (!V2 && b != ntypes) {
      throw std::invalid_argument(
          "TensorData constructor error: If use_zlist is false, b must equal "
          "ntypes.");
    }
    if (V2 && b != 1) {
      throw std::invalid_argument(
          "TensorData constructor error: If use_zlist is true, b must be 1.");
    }

    // Initialize constant dimensions
    dims.b = b;
    dims.k = k;
    dims.m = m;

    // Calculate unique moment dimension d based on m
    if (T_proj_algo == 2) {
      std::vector<int> m2d{0, 1, 4, 9, 16, 25, 36};
      dims.d = m2d[m];
    } else {
      std::vector<int> m2d{0, 1, 4, 10, 20, 35, 56};
      dims.d = m2d[m];
    }
  }

  void allocate_fixed_size_tensors() {
    const size_t ntypes_size = static_cast<size_t>(ntypes);
    const size_t n_eltij_pos = ntypes_size * ntypes_size;
    const size_t n_static_ints = n_eltij_pos + ntypes_size + ntypes_size;

    static_int_pool.grow(n_static_ints);
    static_int_pool.reset();
    eltij_pos = static_int_pool.request(n_eltij_pos);
    elt_a_offset = static_int_pool.request(ntypes_size);
    numbers = static_int_pool.request(ntypes_size);

    fixed_size_tensors_initialized = true;
  }

  auto get_n_scalars_required(bool require_vatom) const {
    size_t n_scalars = 0;
#if defined(TENSORMD_DEBUG)
    (void)require_vatom;
#endif
    const size_t a = static_cast<size_t>(dims.a);
    const size_t b = static_cast<size_t>(dims.b);
    const size_t c = static_cast<size_t>(dims.c);
    const size_t d = static_cast<size_t>(dims.d);
    const size_t k = static_cast<size_t>(dims.k);
    const size_t m = static_cast<size_t>(dims.m);
    const auto a_b_c_1 = a * b * c;
    const auto a_b_c_3 = a_b_c_1 * 3;
    const auto a_b_c_d = a_b_c_1 * d;
    const auto a_b_c_k = a_b_c_1 * k;
    const auto a_b_k_d = a * b * k * d;
    const auto a_b_k_m = a * b * k * m;

    n_scalars += a_b_c_1;  // R
    n_scalars += a_b_c_3;  // drdrx

#if defined(TENSORMD_DEBUG)
    n_scalars += a_b_c_3;  // Fr
    if (STRATEGY == Strategy::Exact) {
      n_scalars += a_b_c_d;  // M
      n_scalars += a_b_c_1;  // sij
      n_scalars += a_b_c_1;  // dsij
      n_scalars += a_b_c_k;  // H
      n_scalars += a_b_c_k;  // dHdr
      n_scalars += a_b_k_d;  // P
      n_scalars += a_b_k_m;  // G
      n_scalars += a_b_k_m;  // dEdG
      n_scalars += a_b_k_d;  // W
      n_scalars += a_b_c_k;  // U
      n_scalars += a_b_c_d;  // V
      n_scalars += a_b_c_k;  // Up
      n_scalars += a_b_c_1;  // dHdrp
      n_scalars += a_b_c_3;  // Fa
    } else if (STRATEGY == Strategy::Baseline) {
      n_scalars += a_b_c_d;  // M
      n_scalars += a_b_c_k;  // H
      n_scalars += a_b_c_k;  // dHdr
      n_scalars += a_b_k_d;  // P
      n_scalars += a_b_k_m;  // G
      n_scalars += a_b_k_m;  // dEdG
      n_scalars += a_b_k_d;  // W
      n_scalars += a_b_c_d;  // U
      n_scalars += a_b_c_d;  // V
      n_scalars += a_b_c_3;  // Fa
    } else if (STRATEGY == Strategy::Speed) {
      if (TENSORMD_USE_FP32_ATLAS) {
        n_scalars += 0;
      } else {
        n_scalars += a_b_k_d;  // P
      }
      n_scalars += a_b_k_m;  // G
      n_scalars += a_b_k_m;  // dEdG
    } else if (STRATEGY == Strategy::Production) {
      n_scalars += a_b_k_m;  // G
      n_scalars += a_b_k_m;  // dEdG
    } else if (STRATEGY == Strategy::Scale) {
      n_scalars += 0;
    }
#else
    if (STRATEGY == Strategy::Exact) {
      n_scalars += a_b_c_d;                     // M
      n_scalars += a_b_c_k;                     // H, U, Up
      n_scalars += a_b_c_k;                     // dHdr
      n_scalars += a_b_k_m;                     // G, dEdG
      n_scalars += std::max(a_b_k_d, a_b_c_d);  // P, V
      n_scalars += a_b_c_1;                     // sij
      n_scalars += a_b_c_1;                     // dsij, dHdrp
      n_scalars += a_b_c_3;                     // Fa
      n_scalars += a_b_c_3;                     // Fr
      n_scalars += a_b_k_d;                     // W
    } else if (STRATEGY == Strategy::Baseline) {
      n_scalars += a_b_c_d;                     // M
      n_scalars += std::max(a_b_c_k, a_b_c_d);  // H, U
      n_scalars += a_b_k_m;                     // G, dEdG
      n_scalars += a_b_c_k;                     // dHdr
      n_scalars += std::max(a_b_k_d, a_b_c_d);  // P, V
      n_scalars += a_b_c_3;                     // Fa
      n_scalars += a_b_c_3;                     // Fr
      n_scalars += a_b_k_d;                     // W
    } else if (STRATEGY == Strategy::Speed) {
      if (TENSORMD_USE_FP32_ATLAS) {
        n_scalars += 0;
      } else {
        n_scalars += a_b_k_d;  // P
      }
      n_scalars += a_b_k_m;  // G, dEdG
      if (require_vatom) {
        n_scalars += a_b_c_3;  // Fr
      }
    } else if (STRATEGY == Strategy::Production) {
      n_scalars += a_b_k_m;  // G, dEdG
      if (require_vatom) {
        n_scalars += a_b_c_3;  // Fr
      }
    } else if (STRATEGY == Strategy::Scale) {
      if (require_vatom) {
        n_scalars += a_b_c_3;  // Fr
      }
    }
#endif

    return n_scalars;
  }

  auto get_n_floats_required() const {
    size_t n_floats = 0;
    if constexpr (std::is_same_v<Scalar, double>) {
      if (TENSORMD_USE_FP32_ATLAS &&
          (STRATEGY == Strategy::Speed || STRATEGY == Strategy::Production)) {
        const size_t a = static_cast<size_t>(dims.a);
        const size_t b = static_cast<size_t>(dims.b);
        const size_t d = static_cast<size_t>(dims.d);
        const size_t k = static_cast<size_t>(dims.k);
        const auto a_b_d_k = a * b * d * k;
        n_floats += a_b_d_k;  // P
      }
    }
    return n_floats;
  }

  // Calculate the total number of integers required for the dynamic tensors
  auto get_n_integers_required() const {
    size_t n_integers = 0;
    const size_t a = static_cast<size_t>(dims.a);
    const size_t b = static_cast<size_t>(dims.b);
    const size_t c = static_cast<size_t>(dims.c);
    const auto a_b_c_1 = a * b * c;
    const auto a_b_c_2 = a_b_c_1 * 2;

    n_integers += a_b_c_2;  // ijlist
    if (dims.b > 1) {
      n_integers += a * 2;  // i2a, a2i
    } else if (V2) {
      if (STRATEGY == Strategy::Exact) {
        n_integers += a_b_c_2;  // zijlist
      }
      n_integers += a;  // tlist
    }
    n_integers += a_b_c_1;  // tijlist
    return n_integers;
  }

  // Resize the memory pool if needed
  void resize(int nlocal, int neigh_max, bool require_vatom) {
    if (!fixed_size_tensors_initialized) {
      throw std::runtime_error(
          "allocate_fixed_size_tensors() should be called before resize.");
    }
    // 1. Update Dimensions
    dims.a = nlocal;
    dims.c = neigh_max;

    // 2. Calculate Total Sizes
    const size_t n_scalars = get_n_scalars_required(require_vatom);
    const size_t n_ints = get_n_integers_required();
    const size_t n_floats = get_n_floats_required();

    // 3. Grow Pools
    scalar_pool.grow(n_scalars);
    int_pool.grow(n_ints);
    float_pool.grow(n_floats);

    // 4. Reset & Assign Pointers
    const size_t a = static_cast<size_t>(dims.a);
    const size_t b = static_cast<size_t>(dims.b);
    const size_t c = static_cast<size_t>(dims.c);
    const size_t d = static_cast<size_t>(dims.d);
    const size_t k = static_cast<size_t>(dims.k);
    const size_t m = static_cast<size_t>(dims.m);
    const auto a_b_c_1 = a * b * c;
    const auto a_b_c_2 = a_b_c_1 * 2;
    const auto a_b_c_3 = a_b_c_1 * 3;
    const auto a_b_c_d = a_b_c_1 * d;
    const auto a_b_c_k = a_b_c_1 * k;
    const auto a_b_k_d = a * b * k * d;
    const auto a_b_k_m = a * b * k * m;

    scalar_pool.reset();
    int_pool.reset();
    float_pool.reset();

    ijlist = nullptr;
    i2a = nullptr;
    a2i = nullptr;
    zijlist = nullptr;
    tijlist = nullptr;
    tlist = nullptr;

    R = nullptr;
    drdrx = nullptr;
    M = nullptr;
    sij = nullptr;
    dsij = nullptr;
    H = nullptr;
    dHdr = nullptr;
    P = nullptr;
    G = nullptr;
    dEdG = nullptr;
    W = nullptr;
    U = nullptr;
    Up = nullptr;
    dHdrp = nullptr;
    V = nullptr;
    Fr = nullptr;
    Fa = nullptr;

    R = scalar_pool.request(a_b_c_1);
    drdrx = scalar_pool.request(a_b_c_3);

#if defined(TENSORMD_DEBUG)
    // The debug mode will keep all intermediate tensors for validation and
    // testing purposes.
    Fr = scalar_pool.request(a_b_c_3);
    if (STRATEGY == Strategy::Exact) {
      M = scalar_pool.request(a_b_c_d);
      H = scalar_pool.request(a_b_c_k);
      dHdr = scalar_pool.request(a_b_c_k);
      P = scalar_pool.request(a_b_k_d);
      U = scalar_pool.request(a_b_c_k);
      V = scalar_pool.request(a_b_c_d);
      Up = scalar_pool.request(a_b_c_k);
      dHdrp = scalar_pool.request(a_b_c_1);
      sij = scalar_pool.request(a_b_c_1);
      dsij = scalar_pool.request(a_b_c_1);
      Fa = scalar_pool.request(a_b_c_3);
      W = scalar_pool.request(a_b_k_d);
      dEdG = scalar_pool.request(a_b_k_m);
      G = scalar_pool.request(a_b_k_m);
    } else if (STRATEGY == Strategy::Baseline) {
      M = scalar_pool.request(a_b_c_d);
      H = scalar_pool.request(a_b_c_k);
      dHdr = scalar_pool.request(a_b_c_k);
      P = scalar_pool.request(a_b_k_d);
      U = scalar_pool.request(a_b_c_d);
      V = scalar_pool.request(a_b_c_d);
      Fa = scalar_pool.request(a_b_c_3);
      W = scalar_pool.request(a_b_k_d);
      G = scalar_pool.request(a_b_k_m);
      dEdG = scalar_pool.request(a_b_k_m);
    } else if (STRATEGY == Strategy::Speed) {
      // M, H, dHdr, U and V are all evaluated on-the-fly
      if (TENSORMD_USE_FP32_ATLAS) {
        P = reinterpret_cast<Scalar*>(float_pool.request(a_b_k_d));
      } else {
        P = scalar_pool.request(a_b_k_d);
      }
      W = P;
      G = scalar_pool.request(a_b_k_m);
      dEdG = scalar_pool.request(a_b_k_m);
    } else if (STRATEGY == Strategy::Production) {
      // M, H, dHdr, U and V are all evaluated on-the-fly
      // P, W are also evaluated on-the-fly
      G = scalar_pool.request(a_b_k_m);
      dEdG = scalar_pool.request(a_b_k_m);
    } else if (STRATEGY == Strategy::Scale) {
      // No intermediate tensors are stored for this strategy
    }
#else
    if (STRATEGY == Strategy::Exact) {
      M = scalar_pool.request(a_b_c_d);
      H = scalar_pool.request(a_b_c_k);
      dHdr = scalar_pool.request(a_b_c_k);
      P = scalar_pool.request(std::max(a_b_k_d, a_b_c_d));
      G = scalar_pool.request(a_b_k_m);
      dEdG = G;
      U = H;
      V = P;
      Up = U;
      dHdrp = scalar_pool.request(a_b_c_1);
      sij = scalar_pool.request(a_b_c_1);
      dsij = dHdrp;
      Fa = scalar_pool.request(a_b_c_3);
      Fr = scalar_pool.request(a_b_c_3);
      W = scalar_pool.request(a_b_k_d);
    } else if (STRATEGY == Strategy::Baseline) {
      M = scalar_pool.request(a_b_c_d);
      H = scalar_pool.request(std::max(a_b_c_k, a_b_c_d));
      dHdr = scalar_pool.request(a_b_c_k);
      P = scalar_pool.request(std::max(a_b_k_d, a_b_c_d));
      G = scalar_pool.request(a_b_k_m);
      dEdG = G;
      U = H;
      V = P;
      Fa = scalar_pool.request(a_b_c_3);
      Fr = scalar_pool.request(a_b_c_3);
      W = scalar_pool.request(a_b_k_d);
    } else if (STRATEGY == Strategy::Speed) {
      // M, H, dHdr, U and V are all evaluated on-the-fly
      if (TENSORMD_USE_FP32_ATLAS) {
        P = reinterpret_cast<Scalar*>(float_pool.request(a_b_k_d));
      } else {
        P = scalar_pool.request(a_b_k_d);
      }
      G = scalar_pool.request(a_b_k_m);
      dEdG = G;
      W = P;
      if (require_vatom) {
        Fr = scalar_pool.request(a_b_c_3);
      } else {
        Fr = drdrx;
      }
    } else if (STRATEGY == Strategy::Production) {
      // M, H, dHdr, U and V are all evaluated on-the-fly
      // P, W are also evaluated on-the-fly
      G = scalar_pool.request(a_b_k_m);
      dEdG = G;
      if (require_vatom) {
        Fr = scalar_pool.request(a_b_c_3);
      } else {
        Fr = drdrx;
      }
    } else if (STRATEGY == Strategy::Scale) {
      if (require_vatom) {
        Fr = scalar_pool.request(a_b_c_3);
      } else {
        Fr = drdrx;
      }
    }
#endif

    ijlist = int_pool.request(a_b_c_2);
    if (V2) {
      if (STRATEGY == Strategy::Exact) {
        zijlist = int_pool.request(a_b_c_2);
      }
      tlist = int_pool.request(a);
    } else if (dims.b > 1) {
      i2a = int_pool.request(a);
      a2i = int_pool.request(a);
    }
    tijlist = int_pool.request(a_b_c_1);
  }

  [[nodiscard]] double get_memory_usage() const {
    double usage = scalar_pool.get_memory_usage();
    usage += int_pool.get_memory_usage();
    usage += float_pool.get_memory_usage();
    usage += static_int_pool.get_memory_usage();
    return usage;
  }

 private:
  MemoryPool<Scalar> scalar_pool;
  MemoryPool<int> int_pool;
  MemoryPool<float> float_pool;  // For mixed-precision
  MemoryPool<int> static_int_pool;
  bool V2;
  bool fixed_size_tensors_initialized = false;
};

class LAMMPSData {
 public:
  Device device;
  int inum = 0;
  int numneigh_max = 0;
  int cmax = 0;
  int imax = 0;
  int nlocal = 0;
  int nghost = 0;
  int nall = 0;
  int* ilist = nullptr;
  int* numneigh = nullptr;
  double cutneighmax = 0.0;
  double* x = nullptr;
  double* f = nullptr;
  double* vatom = nullptr;

  // CPU pointer to neighbor list (array of pointers)
  int** firstneigh = nullptr;

  // Contiguous storage for all neighbor lists for GPU
  // The memory of this pointer is managed by GpuNeighborListWorkspace
  int* firstneigh_flat = nullptr;

  // Domain bounds (for building neighbor lists on GPU)
  double subdomain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // A counter to track how many timesteps have passed since the last neighbor
  // list build.
  int neighlist_ago = 0;

  // type stores LAMMPS atom types
  int* type = nullptr;

  // fmap maps LAMMPS atom types to internal atom types
  int* fmap = nullptr;

  // ntypes is lammps->atom->ntypes
  int ntypes = 0;

  LAMMPSData(Device dev)
      : device(dev),
        int_pool(dev, "LAMMPSData::Pool::integer", true),
        double_pool(dev, "LAMMPSData::Pool::double", true),
        static_int_pool(dev, "LAMMPSData::Pool::static_integer", false) {}

  void alloc_fixed_size_arrays(int ntypes_) {
    if (ntypes_ < 0) {
      throw std::invalid_argument(
          "LAMMPSData::alloc_fixed_size_arrays() received a negative ntypes.");
    }

    ntypes = ntypes_;
    const size_t n_fmap = static_cast<size_t>(ntypes) + 1;
    static_int_pool.grow(n_fmap);
    static_int_pool.reset();
    fmap = static_int_pool.request(n_fmap);
  }

  // Calculate the total number of doubles required for mirroring LAMMPS x, f
  // and vatom arrays on device.
  // Zero atoms on some ranks are possible (e.g. in shock simulations)
  // Only nall = nlocal + nghost needs mirroring here.
  [[nodiscard]] auto get_n_doubles_required(bool require_vatom) const {
    size_t n_doubles = 0;
    if (device == Device::GPU) {
      const size_t nall_size = static_cast<size_t>(nall);
      n_doubles += nall_size * 3;  // x
      n_doubles += nall_size * 3;  // f
      if (require_vatom) {
        n_doubles += nall_size * 6;  // vatom
      }
    }
    return n_doubles;
  }

  // Calculate the total number of integers required for mirroring LAMMPS
  // neighbor list arrays on device. fmap is not included because it is only
  // used for CPU kernels.
  [[nodiscard]] auto get_n_integers_required() const {
    size_t n_integers = 0;
    if (device == Device::GPU) {
      const size_t inum_size = static_cast<size_t>(inum);
      const size_t nlocal_size = static_cast<size_t>(nlocal);
      const size_t nall_size = static_cast<size_t>(nall);
      n_integers += inum_size;  // ilist
      n_integers += nlocal_size;  // numneigh
      n_integers += nall_size;  // type
    }
    return n_integers;
  }

  void set_domain_and_neighbor(int ago, double cutneigh, const double sublo[3],
                               const double subhi[3]) {
    this->neighlist_ago = ago;
    this->cutneighmax = cutneigh;
    this->subdomain[0] = sublo[0] - cutneigh;
    this->subdomain[1] = subhi[0] + cutneigh;
    this->subdomain[2] = sublo[1] - cutneigh;
    this->subdomain[3] = subhi[1] + cutneigh;
    this->subdomain[4] = sublo[2] - cutneigh;
    this->subdomain[5] = subhi[2] + cutneigh;
  }

  // Resize the memory pool if needed
  void resize(int inum_, int imax_, int nlocal_, int nghost_, bool require_vatom) {
    if (inum_ < 0 || imax_ < 0 || nlocal_ < 0 || nghost_ < 0) {
      throw std::invalid_argument(
          "LAMMPSData::resize() received a negative dimension.");
    }
    this->inum = inum_;
    this->imax = imax_;
    this->nlocal = nlocal_;
    this->nghost = nghost_;
    this->nall = nlocal_ + nghost_;

    if (this->static_int_pool.get_capacity() == 0) {
      throw std::runtime_error(
          "alloc_fixed_size_arrays() must be called before resize() to set up "
          "static arrays like fmap.");
    }

    if (device == Device::GPU) {
      const size_t n_doubles = get_n_doubles_required(require_vatom);
      const size_t n_integers = get_n_integers_required();

      double_pool.grow(n_doubles);
      int_pool.grow(n_integers);
      double_pool.reset();
      int_pool.reset();

      x = double_pool.request(static_cast<size_t>(this->nall) * 3);
      f = double_pool.request(static_cast<size_t>(this->nall) * 3);
      vatom = nullptr;
      if (require_vatom) {
        vatom = double_pool.request(static_cast<size_t>(this->nall) * 6);
      }
      ilist = int_pool.request(static_cast<size_t>(this->inum));
      numneigh = int_pool.request(static_cast<size_t>(this->nlocal));
      type = int_pool.request(static_cast<size_t>(this->nall));
    }
  }

  [[nodiscard]] double get_memory_usage() const {
    return int_pool.get_memory_usage() + double_pool.get_memory_usage() +
           static_int_pool.get_memory_usage();
  }


 private:
  MemoryPool<int> int_pool;
  MemoryPool<double> double_pool;
  MemoryPool<int> static_int_pool;  // For static arrays like fmap
};

// Activation Function Types
enum class ActivationType {
  ReLU = 0,
  Softplus = 1,
  Tanh = 2,
  Squareplus = 3,
  Silu = 4,
  Sigmoid = 5
};

inline ActivationType get_activation_type(const std::string& name) {
  std::string actfn = name;
  std::transform(actfn.begin(), actfn.end(), actfn.begin(), tolower);
  if (actfn == "relu") {
    return ActivationType::ReLU;
  } else if (actfn == "softplus") {
    return ActivationType::Softplus;
  } else if (actfn == "tanh") {
    return ActivationType::Tanh;
  } else if (actfn == "squareplus") {
    return ActivationType::Squareplus;
  } else if (actfn == "silu") {
    return ActivationType::Silu;
  } else {
    return ActivationType::Sigmoid;
  }
}

inline ActivationType get_activation_type(int actfn) {
  if (actfn == 0) {
    return ActivationType::ReLU;
  } else if (actfn == 1) {
    return ActivationType::Softplus;
  } else if (actfn == 2) {
    return ActivationType::Tanh;
  } else if (actfn == 3) {
    return ActivationType::Squareplus;
  } else if (actfn == 4) {
    return ActivationType::Silu;
  } else {
    return ActivationType::Sigmoid;
  }
}

}  // namespace TENSORMD

#endif  // TENSORMD_DATA_TYPES_H
