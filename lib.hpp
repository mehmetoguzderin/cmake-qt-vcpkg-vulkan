#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Lib {
  std::string name;
  std::unordered_map<std::string, std::string> data_map;
  std::vector<std::string> data_vector;

  void print_help();
};