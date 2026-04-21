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

#ifndef TENSORMD_CONSTANTS_H
#define TENSORMD_CONSTANTS_H

#include <map>
#include <string>

#ifndef TENSORMD_MAX_NSPECIES
#define TENSORMD_MAX_NSPECIES 16
#endif

#ifndef NEIGHMASK
#define NEIGHMASK 0x1FFFFFFF
#endif

#define PAIR_KEY(Zi, Zj) ((Zi * 100 + Zj))

namespace TENSORMD {

namespace Profiling {
enum TimedKernel {
  Setup,
  get_exact_cmax,
  neighbor_list,
  sync_lammps_data,
  setup_tensors,
  Descriptor,
  interpolate,
  calc_sij,
  forward,
  apply_sij,
  v2_embed_input,
  v2_embed_grad,
  calc_P,
  calc_P_fused,
  calc_G,
  Energy,
  head_forward,
  head_backward,
  head_fused,
  head_hybrid,
  v1_head_ops,
  Gradients,
  calc_W,
  fused_forces,
  RadialForces,
  calc_U,
  calc_Up,
  backward,
  calc_Fr,
  AngularForces,
  calc_V,
  calc_Fa,
  SumForces,
  Overall,
  Null,
  TheOneKernel,
};
}

// Maximum number of kernels for profiling/timing purposes
static constexpr int N_TIMED_KERNELS = 99;

// The size of the pair_map array.
static constexpr int PAIR_MAP_SIZE = 10000;

// Mapping between atomic numbers and symbols
static const std::map<int, std::string> AtomicNumberToSymbol = {
    {0, "X"},   {1, "H"},   {2, "He"},  {3, "Li"},  {4, "Be"},  {5, "B"},
    {6, "C"},   {7, "N"},   {8, "O"},   {9, "F"},   {10, "Ne"}, {11, "Na"},
    {12, "Mg"}, {13, "Al"}, {14, "Si"}, {15, "P"},  {16, "S"},  {17, "Cl"},
    {18, "Ar"}, {19, "K"},  {20, "Ca"}, {21, "Sc"}, {22, "Ti"}, {23, "V"},
    {24, "Cr"}, {25, "Mn"}, {26, "Fe"}, {27, "Co"}, {28, "Ni"}, {29, "Cu"},
    {30, "Zn"}, {31, "Ga"}, {32, "Ge"}, {33, "As"}, {34, "Se"}, {35, "Br"},
    {36, "Kr"}, {37, "Rb"}, {38, "Sr"}, {39, "Y"},  {40, "Zr"}, {41, "Nb"},
    {42, "Mo"}, {43, "Tc"}, {44, "Ru"}, {45, "Rh"}, {46, "Pd"}, {47, "Ag"},
    {48, "Cd"}, {49, "In"}, {50, "Sn"}, {51, "Sb"}, {52, "Te"}, {53, "I"},
    {54, "Xe"}, {55, "Cs"}, {56, "Ba"}, {57, "La"}, {58, "Ce"}, {59, "Pr"},
    {60, "Nd"}, {61, "Pm"}, {62, "Sm"}, {63, "Eu"}, {64, "Gd"}, {65, "Tb"},
    {66, "Dy"}, {67, "Ho"}, {68, "Er"}, {69, "Tm"}, {70, "Yb"}, {71, "Lu"},
    {72, "Hf"}, {73, "Ta"}, {74, "W"},  {75, "Re"}, {76, "Os"}, {77, "Ir"},
    {78, "Pt"}, {79, "Au"}, {80, "Hg"}, {81, "Tl"}, {82, "Pb"}, {83, "Bi"},
    {84, "Po"}, {85, "At"}, {86, "Rn"}, {87, "Fr"}, {88, "Ra"}, {89, "Ac"},
    {90, "Th"}, {91, "Pa"}, {92, "U"},  {93, "Np"}, {94, "Pu"}, {95, "Am"}};

// Reverse mapping from symbols to atomic numbers
static const std::map<std::string, int> AtomicSymbolToNumber = {
    {"X", 0},   {"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},
    {"C", 6},   {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10}, {"Na", 11},
    {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},  {"S", 16},  {"Cl", 17},
    {"Ar", 18}, {"K", 19},  {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23},
    {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29},
    {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
    {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39},  {"Zr", 40}, {"Nb", 41},
    {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47},
    {"Cd", 48}, {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53},
    {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59},
    {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65},
    {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71},
    {"Hf", 72}, {"Ta", 73}, {"W", 74},  {"Re", 75}, {"Os", 76}, {"Ir", 77},
    {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83},
    {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89},
    {"Th", 90}, {"Pa", 91}, {"U", 92},  {"Np", 93}, {"Pu", 94}, {"Am", 95}};
}  // namespace TENSORMD

#endif
