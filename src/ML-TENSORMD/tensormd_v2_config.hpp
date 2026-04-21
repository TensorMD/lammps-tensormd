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

#ifndef TENSORMD_V2_CONFIG_HPP
#define TENSORMD_V2_CONFIG_HPP

#include <string>
#include <vector>

#include "cnpy++.hpp"
#include "yaml-cpp/yaml.h"

namespace TENSORMD {

struct MLPEncoderConfig {
  std::vector<int> hidden_dims;
  int r_feat_dim;

  explicit MLPEncoderConfig(const YAML::Node& node) {
    if (!node || !node.IsMap())
      throw std::runtime_error("MLPEncoderConfig: node is not a map");

    auto hd = node["hidden_dims"];
    if (!hd || !hd.IsSequence())
      throw std::runtime_error(
          "MLPEncoderConfig: missing or invalid hidden_dims");
    hidden_dims = hd.as<std::vector<int>>();

    auto rf = node["r_feat_dim"];
    if (!rf) throw std::runtime_error("MLPEncoderConfig: missing r_feat_dim");
    r_feat_dim = rf.as<int>();
  }

  explicit MLPEncoderConfig() { r_feat_dim = 0; }
};

struct NeuralEnergyHeadConfig {
  std::vector<int> hidden_dims;
  std::string actfn;
  bool use_resnet_dt;

  explicit NeuralEnergyHeadConfig(const YAML::Node& node) {
    if (!node || !node.IsMap())
      throw std::runtime_error("NeuralEnergyHeadConfig: node is not a map");
    use_resnet_dt =
        node["use_residual"] ? node["use_residual"].as<bool>() : false;
    actfn = node["activation_fn_str"]
                ? node["activation_fn_str"].as<std::string>()
                : "softplus";
    auto hd = node["hidden_dims"];
    if (!hd || !hd.IsSequence())
      throw std::runtime_error(
          "NeuralEnergyHeadConfig: missing or invalid hidden_dims");
    hidden_dims = hd.as<std::vector<int>>();
  }

  explicit NeuralEnergyHeadConfig() {
    use_resnet_dt = false;
    actfn = "softplus";
  }
};

struct ModelConfig {
  std::vector<int> atomic_numbers;
  std::vector<int> species_chars;
  std::vector<double> masses;
  double r_cut;
  int max_moment;
  int k_dim;
  bool use_compressed_T;
  MLPEncoderConfig mlp;
  NeuralEnergyHeadConfig energy_head;
  std::string active_task_id;

  explicit ModelConfig(const YAML::Node& node) {
    if (!node || !node.IsMap())
      throw std::runtime_error("ModelConfig: root is not a map");

    auto encoder = node["encoder"];
    if (!encoder || !encoder.IsMap())
      throw std::runtime_error("ModelConfig: missing encoder");

    auto tasks = node["tasks"];
    if (!tasks || !tasks.IsSequence() || tasks.size() == 0)
      throw std::runtime_error("ModelConfig: missing tasks");

    mlp = MLPEncoderConfig(encoder["mlp_encoder"]);
    energy_head = NeuralEnergyHeadConfig(tasks[0]["energy_head"]);

    r_cut = encoder["r_cut"].as<double>();
    max_moment = encoder["max_moment"].as<int>();
    k_dim = encoder["k_dim"].as<int>();
    active_task_id = tasks[0]["id"].as<std::string>();
    if (encoder["use_compressed_T"]) {
      use_compressed_T = encoder["use_compressed_T"].as<bool>();
    } else {
      use_compressed_T = true;
    }
  }

  explicit ModelConfig() {
    r_cut = 0.0;
    max_moment = 0;
    use_compressed_T = true;
    k_dim = 0;
    active_task_id = "";
  }
};

// Function to load the configuration from the NPZ map
static ModelConfig load_config_from_npz(cnpypp::npz_t& npz_map) {
  // Check if the metadata key exists
  if (npz_map.find("__metadata_config__") == npz_map.end()) {
    throw std::runtime_error("NPZ file is missing '__metadata_config__' key.");
  }

  cnpypp::NpyArray const& arr = npz_map.find("__metadata_config__")->second;
  const auto* p = arr.data<uint8_t>();
  std::string yaml_string(reinterpret_cast<const char*>(p), arr.num_vals);
  YAML::Node node = YAML::Load(yaml_string);

  // Read the config
  auto config = ModelConfig(node);

  // Additional properties
  auto numbers = npz_map.find("__atomic_numbers__");
  if (numbers != npz_map.end()) {
    for (int i = 0; i < static_cast<int>(numbers->second.num_vals); i++) {
      config.atomic_numbers.push_back(numbers->second.data<int>()[i]);
    }
  }
  auto chars = npz_map.find("__species_chars__");
  if (chars != npz_map.end()) {
    for (int i = 0; i < static_cast<int>(chars->second.num_vals); i++) {
      config.species_chars.push_back(chars->second.data<int>()[i]);
    }
  }
  auto masses = npz_map.find("__masses__");
  if (masses != npz_map.end()) {
    for (int i = 0; i < static_cast<int>(masses->second.num_vals); i++) {
      config.masses.push_back(masses->second.data<double>()[i]);
    }
  }

  return config;
}

}  // namespace TENSORMD

#endif  // TENSORMD_V2_CONFIG_HPP
