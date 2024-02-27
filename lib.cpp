#include "lib.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

NB_MODULE(LibCxx, m) {
  nb::class_<Lib>(m, "Dog")
      .def(nb::init<>())
      .def(nb::init<const std::string &>())
      .def_rw("name", &Lib::name)
      .def("help", &Lib::help);
}
