//====- LowerToAffineLoops.cpp - Partial lowering from Relay to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Relay operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//
#include<iostream>

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// RelayToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as relay functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input a rewriter, an array of memRefOperands corresponding
/// to the operands of the input operation, and the set of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn = function_ref<Value(PatternRewriter &rewriter,
                                           ArrayRef<Value> memRefOperands,
                                           ArrayRef<Value> loopIvs)>;

static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create an empty affine loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  for (auto dim : tensorType.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loop.getBody()->clear();
    loopIvs.push_back(loop.getInductionVar());

    // Terminate the loop body and update the rewriter insertion point to the
    // beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Generate a call to the processing function with the rewriter, the memref
  // operands, and the loop induction variables. This function will return the
  // value to store at the current index.
  Value valueToStore = processIteration(rewriter, operands, loopIvs);
  rewriter.create<AffineStoreOp>(loc, valueToStore, alloc,
                                 llvm::makeArrayRef(loopIvs));

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// RelayToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the BinaryOp. This
          // allows for using the nice named accessors that are generated by the
          // ODS.
          typename BinaryOp::OperandAdaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the inner
          // loop.
          auto loadedLhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
          auto loadedRhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

          // Create the binary operation performed on the loaded values.
          return rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<relay::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<relay::MulOp, MulFOp>;

//===----------------------------------------------------------------------===//
// RelayToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<relay::ConstantOp> {
  using OpRewritePattern<relay::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(relay::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
              0, *std::max_element(valueShape.begin(), valueShape.end())))
       constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RelayToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<relay::ReturnOp> {
  using OpRewritePattern<relay::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(relay::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "relay.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RelayToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(relay::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS.
          relay::TransposeOpOperandAdaptor transposeAdaptor(memRefOperands);
          Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};

struct Conv2dOpLowering : public ConversionPattern {
  Conv2dOpLowering(MLIRContext *ctx)
      : ConversionPattern(relay::Conv2dOp::getOperationName(), 1, ctx) {}
    LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
  auto loc = op->getLoc();

  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto tensorType1 =  (*op->operand_type_begin()).cast<TensorType>();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  //--initialize %0 with 0
  auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
              0, *std::max_element(valueShape.begin(), valueShape.end())))
       constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    MLIRContext* context = rewriter.getContext();
    FloatAttr zero  =  FloatAttr::get(FloatType::getF64(context), 0.);
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, zero), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

  // Create an empty affine loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  SmallVector<Value, 4> loopIvs0;
  for (auto dim : tensorType.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loop.getBody()->clear();
    loopIvs.push_back(loop.getInductionVar());
    loopIvs0.push_back(loop.getInductionVar());

    // Terminate the loop body and update the rewriter insertion point to the
    // beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  SmallVector<Value, 4> loopIvs1;
  for (auto dim : tensorType1.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loop.getBody()->clear();
    loopIvs.push_back(loop.getInductionVar());
    loopIvs1.push_back(loop.getInductionVar());

    // Terminate the loop body and update the rewriter insertion point to the
    // beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  relay::Conv2dOpOperandAdaptor conv2dAdaptor(operands);

          // Transpose the elements by generating a load from the reverse
          // indices.
  auto loadedLhs =
        rewriter.create<AffineLoadOp>(loc,conv2dAdaptor.filter(), loopIvs1);
  // SmallVector<Value, 4> loopIvs2;
  // for(int i=0;i<loopIvs.size();i++)
  // {
  //   Value tmp = rewriter.create<AddIOp>(loc, loopIvs[i], loopIvs1[i]);
  //   loopIvs2.push_back(tmp);
  // }
  //std::cout<<"Affine load 1 succ"<<std::endl;
  AffineExpr d0,d1,d2,d3;
  //std::cout<<"Affine Expr"<<std::endl;
   
  // bindDims(context, d0, d1);
  //std::cout<<"context"<<std::endl;
  SmallVector<AffineExpr, 4> exprs;
  bindDims(context, d0, d1, d2, d3);
  //std::cout<<"bindDims"<<std::endl;
  exprs.push_back(d0+d2);
  exprs.push_back(d1+d3);
  AffineMap map = AffineMap::get(4,0,exprs,context);
  //std::cout<<"AffineMap construct succ"<<std::endl;
  auto loadedRhs =
        rewriter.create<AffineLoadOp>(loc, conv2dAdaptor.input(),map,loopIvs);
  //std::cout<<"Affine Load 2 succ"<<std::endl;
          // Create the binary operation performed on the loaded values.
  auto addLeft = rewriter.create<MulFOp>(loc, loadedLhs, loadedRhs);
  auto addRight = rewriter.create<AffineLoadOp>(loc, alloc,loopIvs0);
  auto valueToStore = rewriter.create<AddFOp>(loc, addLeft, addRight);
  rewriter.create<AffineStoreOp>(loc, valueToStore, alloc,
                                 llvm::makeArrayRef(loopIvs0));

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
  return success();
  };
  
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// RelayToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the relay operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Relay dialect.
namespace {
struct RelayToAffineLoweringPass
    : public PassWrapper<RelayToAffineLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void RelayToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "main")
    return;

  // Verify that the given main has no inputs and results.
  // if (function.getNumArguments() || function.getType().getNumResults()) {
  //   function.emitError("expected 'main' to have 0 inputs and 0 results");
  //   return signalPassFailure();
  // }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();

  // We also define the Relay dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Relay operations that don't want
  // to lower, `relay.print`, as `legal`.
  target.addIllegalDialect<relay::RelayDialect>();
  target.addLegalOp<relay::PrintOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Relay operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, ConstantOpLowering, MulOpLowering,
                  ReturnOpLowering, TransposeOpLowering,Conv2dOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Relay IR (e.g. matmul).
std::unique_ptr<Pass> mlir::relay::createLowerToAffinePass() {
  return std::make_unique<RelayToAffineLoweringPass>();
}
