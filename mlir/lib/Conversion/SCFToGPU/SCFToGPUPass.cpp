//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
// #include "/usr/include/llvm-10/llvm/Support/CommandLine.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINEFORTOGPU
#define GEN_PASS_DEF_CONVERTPARALLELLOOPTOGPU
#define GEN_PASS_DEF_FOOPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public impl::ConvertAffineForToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnOperation() override {
    for (Operation &op : llvm::make_early_inc_range(
             getOperation().getFunctionBody().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      }
    }
      printf("finished affine-for-to-gpu pass\n");
  }
};

struct ParallelLoopToGpuPass
    : public impl::ConvertParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateParallelLoopToGPUPatterns(patterns);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    configureParallelLoopToGPULegality(target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
    finalizeParallelLoopToGPUConversion(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::createAffineForToGPUPass(unsigned numBlockDims, unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}
std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::createAffineForToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

std::unique_ptr<Pass> mlir::createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}

namespace {
struct FooPass : public impl::FooPassBase<FooPass> {
  void runOnOperation() override {
    std::vector<AffineForOp> inputNest;
    Operation * op = getOperation();
    op->walk([&](Operation *inst){
      if (auto forOp = dyn_cast<AffineForOp>(inst)) 
      {
        inputNest.insert(inputNest.begin(), forOp);
    	}
    });

    std::vector<unsigned> permMap;
    unsigned idx = 0;
    for(unsigned i = 0; i < inputNest.size(); i++)
    {
      permMap.push_back(0);
      if(isLoopMemoryParallel(inputNest[i]))
      {
        permMap[i] = idx;
        idx++;
      }
    }
    for(unsigned j = 0; j < inputNest.size(); j++)
    {
      if(!isLoopMemoryParallel(inputNest[j]))
      {
        permMap[j] = idx;
        idx++;
      }
    }
    ArrayRef<unsigned> permMapAR(permMap);
    MutableArrayRef<AffineForOp> inputNestMAR(inputNest);
    if(isValidLoopInterchangePermutation(inputNestMAR, permMapAR))
    {
      permuteLoops(inputNestMAR, permMapAR);
      // printf("input nest");
    }
    else
    {
      op->emitOpError("Invalid Loop Interchange Permutation\n");
    }

    printf("finished foo-pass\n");
  }
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> mlir::createFooPassPass() {
  return std::make_unique<FooPass>();
}


namespace{
  struct ScaleCUDAPipelineOptions : public PassPipelineOptions<ScaleCUDAPipelineOptions> {
    // The structure of these options is the same as those for pass options.
    // Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
    // ListOption<int> exampleListOption{*this, "list-flag-name",
    //                                   llvm::cl::desc("...")};
  };
}

//NOTE: also added a call to this in: llvm-project/mlir/include/mlir/InitAllPasses.h
// and a declaration in: llvm-project/mlir/include/mlir/Conversion/SCFToGPU/SCFToGPUPass.h
void mlir::registerScaleCUDAPipeline() {
    mlir::PassPipelineRegistration<ScaleCUDAPipelineOptions>(
    "scalecuda-pipeline", "Optimize Affine on the GPU dialect", [](OpPassManager &pm, const ScaleCUDAPipelineOptions &opts) {
      pm.addPass(mlir::createFooPassPass());
      pm.addPass(mlir::createAffineForToGPUPass());
    });
}
