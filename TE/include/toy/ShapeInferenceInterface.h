//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_RELAY_SHAPEINFERENCEINTERFACE_H_
#define MLIR_TUTORIAL_RELAY_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace relay {

/// Include the auto-generated declarations.
#include "toy/ShapeInferenceOpInterfaces.h.inc"

} // end namespace relay
} // end namespace mlir

#endif // MLIR_TUTORIAL_RELAY_SHAPEINFERENCEINTERFACE_H_
