#pragma once

#include <nvbench/axis_base.cuh>

#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>

#include <string>
#include <vector>

namespace nvbench
{

struct type_axis final : public axis_base
{
  type_axis(std::string name, std::size_t axis_index)
      : axis_base{std::move(name), axis_type::type}
      , m_input_strings{}
      , m_descriptions{}
      , m_axis_index{axis_index}
  {}

  ~type_axis() final;

  template <typename TypeList>
  void set_inputs();

  /**
   * The index of this axis in the `benchmark`'s `type_axes` type list.
   */
  [[nodiscard]] std::size_t get_axis_index() const { return m_axis_index; }

  /**
   * The index in this axis of the type with the specified `input_string`.
   */
  [[nodiscard]] std::size_t
  get_type_index(const std::string &input_string) const;

private:
  std::size_t do_get_size() const final { return m_input_strings.size(); }
  std::string do_get_input_string(std::size_t i) const final
  {
    return m_input_strings[i];
  }
  std::string do_get_description(std::size_t i) const final
  {
    return m_descriptions[i];
  }

  std::vector<std::string> m_input_strings;
  std::vector<std::string> m_descriptions;
  std::size_t m_axis_index;
};

template <typename TypeList>
void type_axis::set_inputs()
{
  // Need locals for lambda capture...
  auto &input_strings = m_input_strings;
  auto &descriptions  = m_descriptions;
  nvbench::tl::foreach<TypeList>(
    [&input_strings, &descriptions]([[maybe_unused]] auto wrapped_type) {
      using T       = typename decltype(wrapped_type)::type;
      using Strings = nvbench::type_strings<T>;
      input_strings.push_back(Strings::input_string());
      descriptions.push_back(Strings::description());
    });
}

} // namespace nvbench
