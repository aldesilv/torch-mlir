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
  patterns.onOp("Identity", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOp(binder.op, operand);
                  return success();
                });

  patterns.onOp("Max", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  Torch::ValueTensorType resultType;
		  llvm::SmallVector<Value> operands;
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
  patterns.onOp("MaxPool", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  MLIRContext *context = binder.op->getContext();
                  Value constantOne = rewriter.create<Torch::ConstantIntOp>(
                    binder.getLoc(), rewriter.getI64IntegerAttr(1));
                  Value constantZero = rewriter.create<Torch::ConstantIntOp>(
                    binder.getLoc(), rewriter.getI64IntegerAttr(0));

		  SmallString<64> name("torch.onnx.");
                  name.append("kernel_shape");
                  auto attr = binder.op->getAttr(name);

                  if (!attr) {
                    return failure();
                  }

                  auto kernelSizeAttr = dyn_cast<ArrayAttr>(attr);


		  bool ceil_mode;
		  binder.s64BoolAttr(ceil_mode, "ceil_mode", false);

		  name = "torch.onnx.";
                  name.append("strides");
                  attr = binder.op->getAttr(name);

		  SmallVector<Value, 1> strides;
                  if (!attr) {
                    for (int i = 0; i < kernelSizeAttr.size(); i++) {
                      strides.push_back(constantOne);
		    }
                  } else {
                    auto stridesAttr = dyn_cast<ArrayAttr>(attr);
                    for (int i = 0; i < stridesAttr.size(); i++) {
                      auto stridesI = llvm::cast<IntegerAttr>(stridesAttr[i]).getInt();
                      strides.push_back(rewriter.create<Torch::ConstantIntOp>(
                        binder.getLoc(), rewriter.getI64IntegerAttr(stridesI)));
                    }
		  }
		  Value stridesList = rewriter.create< Torch::PrimListConstructOp>(
                    binder.getLoc(), Torch::ListType::get(Torch::IntType::get(context)), strides);

		  SmallVector<Value, 1> kernelSize;
		  for (int i = 0; i < kernelSizeAttr.size(); i++) {
	            auto kernelSizeI = llvm::cast<IntegerAttr>(kernelSizeAttr[i]).getInt();
                    kernelSize.push_back(rewriter.create<Torch::ConstantIntOp>(
                      binder.getLoc(), rewriter.getI64IntegerAttr(kernelSizeI)));
		  }
		  Value kernelSizeList = rewriter.create< Torch::PrimListConstructOp>(
                    binder.getLoc(), Torch::ListType::get(Torch::IntType::get(context)), kernelSize);

		  name = "torch.onnx.";
                  name.append("pads");
                  attr = binder.op->getAttr(name);

                  SmallVector<Value, 1> pads;
                  if (!attr) {
                    for (int i = 0; i < kernelSizeAttr.size(); i++) {
                      pads.push_back(constantZero);
                    }
                  } else {
	            auto padsAttr = dyn_cast<ArrayAttr>(attr);
		    for (int i = 0; i < padsAttr.size(); i++) {
                      auto padI = llvm::cast<IntegerAttr>(padsAttr[i]).getInt();
                      pads.push_back(rewriter.create<Torch::ConstantIntOp>(
                        binder.getLoc(), rewriter.getI64IntegerAttr(padI)));
                    }
		  }
	          Value paddingList = rewriter.create< Torch::PrimListConstructOp>(
                    binder.getLoc(), Torch::ListType::get(Torch::IntType::get(context)), pads);

		  name = "torch.onnx.";
                  name.append("dilations");
                  attr = binder.op->getAttr(name);

		  SmallVector<Value, 1> dilation;
		  if (!attr) {
                    for (int i = 0; i < kernelSizeAttr.size(); i++) {
                      dilation.push_back(constantOne);
                    }
                  } else {
                    auto dilationAttr = dyn_cast<ArrayAttr>(attr);
                    for (int i = 0; i < dilationAttr.size(); i++) {
                      auto dilationI = llvm::cast<IntegerAttr>(dilationAttr[i]).getInt();
                      dilation.push_back(rewriter.create<Torch::ConstantIntOp>(
                        binder.getLoc(), rewriter.getI64IntegerAttr(dilationI)));
                    }
                  }
                  Value dilationList = rewriter.create< Torch::PrimListConstructOp>(
                    binder.getLoc(), Torch::ListType::get(Torch::IntType::get(context)), dilation);

		  Value ceil_mode_ = rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceil_mode);
		  
		  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }

		  if (kernelSizeAttr.size() == 3) {
                    rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dOp>(
                      binder.op, resultType, operand, kernelSizeList, stridesList,
                      paddingList, dilationList, ceil_mode_);
		    return success();
                  } else if ( kernelSizeAttr.size() == 2) {
		    rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dOp>(
                      binder.op, resultType, operand, kernelSizeList, stridesList,
		      paddingList, dilationList, ceil_mode_);
		    return success();
		  }
		  return failure();
  	        });
  patterns.onOp("Min", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  llvm::SmallVector<Value> operands;
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
  patterns.onOp("Mul", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenMulTensorOp>(
		    binder.op, resultType, lhs, rhs);
		  return success();
		});
  patterns.onOp("MatMul", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
                    binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Pow", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenPowTensorTensorOp>(
                    binder.op, resultType, lhs, rhs);
                  return success();
                });
}
