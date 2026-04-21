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

#ifndef TENSORMD_H
#define TENSORMD_H

#include <cnpy++.hpp>
#include <string>
#include <vector>

#include "tensormd_memory.h"
#include "encoders/encoder.h"
#include "heads/energy_head.h"
#include "tensormd_context.h"
#include "tensormd_timer.h"

namespace TENSORMD {

template <typename Scalar>
class TensorMD {
 public:
  TensorMD(Device dev, int mpiid);
  virtual ~TensorMD();

  // Initialize the model using the given npz file and return the model
  // description string.
  virtual void initialize(cnpypp::npz_t& npz, std::string& desc) = 0;

  // Adjust the calculation strategy based on model settings.
  virtual void adjust_strategy(std::string& msg) = 0;

  // Setup for global
  void setup_global(const int* fmap, int atom_ntypes);
  virtual void fill_fixed_size_tensors(int* eltij_pos) = 0;

  // Setup the encoder interpolation
  void setup_interpolation(std::vector<std::string>& lmp_species, Scalar rmin,
                           Scalar delta);

  // If device is not CPU, we may copy some LAMMPS arrays to device side
  // so that the kernels can access them directly.
  void sync_lammps_data(int* ilist, int inum, int nlocal, int nghost, int imax,
                        int* type, int* fmap, double** x, int* numneigh,
                        int** firstneigh, double** f, double** vatom);

  // Build neighbor list on GPU if needed (ago == 0)
  // cutneigh is LAMMPS neighbor list cutoff, which equals to cutforce + skin
  void build_neighbor_list(int ago, double cutneigh, double sublo[3],
                           double subhi[3]);

  // Calculate the exact cmax for this step.
  virtual int get_exact_cmax();

  // Setup tensor data for each step
  virtual void setup_local(int inum, int cmax, const int* typenums,
                           bool require_vatom);

  // Main Compute
  virtual void compute(int* typenums, double* etemp, bool use_atom_etemp,
                       FreeEnergyResult<double>& result, double** f,
                       double** vatom, double scale);

  // Return the total memory usage (bytes)
  [[nodiscard]] double get_total_memory_usage() const;

  // Return the memory usage breakdown by components (bytes) and device
  // The keys of the map are:
  //    "Device::TensorData"
  //    "Device::LAMMPSData"
  //    "Device::Encoder"
  //    "Device::EnergyHead"
  [[nodiscard]] std::map<std::string, double> get_memory_usages() const;

  // Return True if this is a finite temperature model
  [[nodiscard]] virtual bool is_finite_temperature_model() const {
    if (energy_head == nullptr) return false;
    return energy_head->is_finite_temp_head();
  }

  // Timer APIs
  void switch_timer(bool on) { timer.enabled = on; }
  void print_timer(std::vector<FILE*>& c_streams) {
    timer.accumulate();
    timer.collect_and_print(c_streams, get_memory_usages());
  }

  [[nodiscard]] Scalar get_rcut() const { return this->context->cutforce; }
  [[nodiscard]] std::map<int, std::string> get_species() const {
    return species;
  }
  [[nodiscard]] std::vector<double> get_masses() const { return masses; }
  [[nodiscard]] std::vector<int> get_numbers() const { return numbers; }

 protected:
  TensorMDTimer timer;

  // All the essential tensors and their dimensions are managed here
  TensorMDContext<Scalar>* context = nullptr;

  // Utility function to dump tensors to file for debugging
  // Currently we use environment variable TENSORMD_DUMP_TENSORS to
  // specify which tensors to dump.
  void may_dump_tensors();

  // Atomic numbers of the species
  std::vector<int> numbers;

  // Masses of the species.
  std::vector<double> masses;

  // A map from element type index to element symbol (e.g. 0 -> "Al").
  std::map<int, std::string> species;

  // Components
  Encoder<Scalar>* encoder = nullptr;
  EnergyHead<Scalar>* energy_head = nullptr;

  // These should be implemented in the derived classes
  virtual void compute_energy(const int* typenums, EnergyResult<double>& result,
                              double scale) = 0;
  virtual void compute_free_energy(const int* typenums, double* etemp,
                                   bool use_atom_etemp,
                                   FreeEnergyResult<double>& result,
                                   double scale) = 0;
  
  // Temporary workaround for the unified kernel
  MemoryPool<Scalar> dynamic_pool;
  std::vector<Scalar> h_eatoms;  // Host buffer for eatoms when using unified kernel
  Scalar *d_eatoms = nullptr;
};

}  // namespace TENSORMD

#endif
