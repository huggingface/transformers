// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMPILE_TIME_VERSION_H
#define COMPILE_TIME_VERSION_H

#include <poplar/Graph.hpp>

inline std::string getPluginVersion() {
    // Replace parenthesis and space in version string so
    // we can easily use the results as a variable in a
    // Makefile and on the compiler command line:
    std::string version = poplar::versionString();
    for (char &c : version) {
        if (c == '(' || c == ')' || c == ' ') {
            c = '-';
        }
    }
    return version;
}

#ifdef STATIC_VERSION
static void __attribute__ ((constructor)) shared_object_init() {
  const std::string runtimeVersion = getPluginVersion();
  if (runtimeVersion != STATIC_VERSION) {
    std::cerr << "ERROR: plug-in version mismatch\n"
              << "STATIC VERSION: " << STATIC_VERSION << " RUN-TIME VERSION: " << runtimeVersion << "\n"
              << "Please recompile the custom operators with the sdk that you are using by runing `make' \n";
    exit(1);
  }
}
#endif

#endif
