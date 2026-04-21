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

#ifndef TENSORMD_CONTEXT_H
#define TENSORMD_CONTEXT_H

#include "encoders/encoder_context.h"
#include "heads/neural_head_context.h"
#include "tensormd_types.h"

namespace TENSORMD {

template <typename Scalar>
class TensorMDContext {
 public:
  // --- The constructor and destructor ---
  TensorMDContext(Device dev, int mpiid)
      : device(dev), mpiid(mpiid), lmpdata(dev) {};
  ~TensorMDContext() { delete data; };

  // --- The main data containers for current context ---

  TensorData<Scalar> *data = nullptr;
  LAMMPSData lmpdata;
  const EncoderContext<Scalar> *encoder = nullptr;
  const NeuralHeadContext<Scalar> *head = nullptr;

  // --- Additional context information ---

  Device device;
  int mpiid;
  bool verbose_device_map = false;

  // For building neighbor list
  double cutforcesq = 0.0;
  double cutforce = 0.0;
};

}  // namespace TENSORMD

#endif
