// Copyright 2026 Open Source Robotics Foundation, Inc.
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

#ifndef QC_BUFFER__QC_ERROR_HPP_
#define QC_BUFFER__QC_ERROR_HPP_

#include <stdexcept>
#include <string>

namespace qc_buffer_backend
{

/// \brief Exception type for qc_buffer errors that break the API contract
/// (e.g. an explicit qc allocation request that fails, or a type mismatch).
/// Environmental or scenario limitations that merely prevent zero-copy are
/// handled by graceful CPU fallback, not by throwing.
class QcError : public std::runtime_error
{
public:
  explicit QcError(const std::string & message)
  : std::runtime_error(message) {}
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__QC_ERROR_HPP_
