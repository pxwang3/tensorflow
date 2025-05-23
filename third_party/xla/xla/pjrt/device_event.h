/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_DEVICE_EVENT_H_
#define XLA_PJRT_DEVICE_EVENT_H_

#include "xla/pjrt/pjrt_future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// A device event occurs (potentially) on a device. It can be waited on
// directly or passed between APIs which may be able to handle these events
// directly.
class PjRtDeviceEvent : public tsl::ReferenceCounted<PjRtDeviceEvent> {
 public:
  virtual ~PjRtDeviceEvent() = default;

  // Converts a device-event into a future.
  virtual PjRtFuture<> GetReadyFuture() = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_H_
