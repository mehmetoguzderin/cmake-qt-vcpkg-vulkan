#include "python.hpp"

namespace nb = nanobind;

NB_MAKE_OPAQUE(std::unordered_map<std::string, std::string>);
NB_MAKE_OPAQUE(std::vector<std::string>);

NB_MODULE(PythonCxx, m) {
  nb::bind_map<std::unordered_map<std::string, std::string>>(
      m, "UnorderedMapFromStdStringToStdString");
  nb::bind_vector<std::vector<std::string>>(m, "VectorOfStdString");

  nb::class_<Lib>(m, "Lib")
      .def(nb::init<>())
      .def_rw("name", &Lib::name)
      .def_rw("data_map", &Lib::data_map)
      .def_rw("data_vector", &Lib::data_vector)
      .def("print_help", &Lib::print_help);
}
