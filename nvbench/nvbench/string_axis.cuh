#pragma once

#include <nvbench/axis_base.cuh>

#include <nvbench/types.cuh>

#include <vector>

namespace nvbench
{

struct string_axis final : public axis_base
{
  explicit string_axis(std::string name)
      : axis_base{std::move(name), axis_type::string}
      , m_values{}
  {}

  ~string_axis() final;

  void set_inputs(std::vector<std::string> inputs)
  {
    m_values = std::move(inputs);
  }
  [[nodiscard]] const std::string &get_value(std::size_t i) const
  {
    return m_values[i];
  }

private:
  std::size_t do_get_size() const final { return m_values.size(); }
  std::string do_get_input_string(std::size_t i) const final
  {
    return m_values[i];
  }
  std::string do_get_description(std::size_t i) const final { return {}; }

  std::vector<std::string> m_values;
};

} // namespace nvbench
