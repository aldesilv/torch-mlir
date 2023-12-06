//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

// Simple rewrites for the default domain.
// See: https://onnx.ai/onnx/operators/
// For operators that are effectively version invariant, we register with
// sinceVersion==1. We interpret this to include the following spec
// diffs that are irrelevant to this level of lowering:
//   * Supported element types.
//   * Limited broadcasting to full broadcasting support.
//
// There are a lot of spec revisions that basically generalized elementwise
// to be more normal and a direct translation vs a special case. This
// results in a lot of ONNX test cases that all reduce to the exact same
// thing here, so we simplify.
void mlir::torch::onnx_c::populateDefaultDomainGtoP(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp("Max", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  Torch::ValueTensorType resultType;
		  llvm::SmallVector<Value, 4> operands;
		  if (binder.tensorOperandsList(operands) ||
                      binder.tensorResultType(resultType) ||
		      operands.size() == 0) {
                    return failure();
		  }
		  Value result = operands[0];
		  for (int i = 1; i < operands.size(); i++) {
		    result = rewriter.create<Torch::AtenMaximumOp>(
		               binder.getLoc(), resultType, result, operands[i]);
		  }
		  rewriter.replaceOp(
                    binder.op, result.getDefiningOp());
		  return success();
                });
  patterns.onOp("Min", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  llvm::SmallVector<Value, 4> operands;
                  if (binder.tensorOperandsList(operands) ||
                      binder.tensorResultType(resultType) ||
                      operands.size() == 0) {
                    return failure();
                  }
                  Value result = operands[0];
                  for (int i = 1; i < operands.size(); i++) {
                    result = rewriter.create<Torch::AtenMinimumOp>(
                               binder.getLoc(), resultType, result, operands[i]);
                  }
                  rewriter.replaceOp(
                    binder.op, result.getDefiningOp());
                  return success();
		});
  patterns.onOp("Less", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLtTensorOp>(
                      binder.op, resultType, lhs, rhs);
		  return success();
                });
  patterns.onOp("Log", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLogOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Neg", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNegOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Not", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNegOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Or", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenBitwiseOrTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });

}
