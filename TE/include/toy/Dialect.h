//===- Dialect.h - Dialect definition for the Relay IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Relay language.
// See docs/Tutorials/Relay/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_RELAY_DIALECT_H_
#define MLIR_TUTORIAL_RELAY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffects.h"
#include "toy/ShapeInferenceInterface.h"

namespace mlir {
namespace relay {

/// This is the definition of the Relay dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class RelayDialect : public mlir::Dialect {
public:
  explicit RelayDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "relay"; }
};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

} // end namespace toy
} // end namespace mlir

#endif // MLIR_TUTORIAL_RELAY_DIALECT_H_
