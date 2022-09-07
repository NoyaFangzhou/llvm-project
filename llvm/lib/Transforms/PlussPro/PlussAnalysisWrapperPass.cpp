//
// Created by noya-fangzhou on 11/5/21.
//
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
#include "SamplerCodeGen.h"
#include "OldSamplerCodeGen.h"
#include "SamplerCodeGenUtils.h"
#include <string>

#define DEBUG_TYPE "pluss"

using namespace llvm;
using namespace std;

namespace PlussWrapperPass {

struct PlussWrapperPass : public FunctionPass {

  static char ID;
  PlussWrapperPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
#if 0
    if (FunctionName.empty()) {
      // no function name given, all function is a target
    } else {
      dbgs() << FunctionName << "\n";
      // the function name is given, only the given function will be analyzed
      if (F.getName().str().find(FunctionName) != string::npos) {
      } else {
        NumFunctionSkipped++;
        LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
      }
    }
#endif
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addPreservedID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addPreservedID(LCSSAID);
    AU.addRequired<SamplerCodeGen::SamplerCodeGenWrapperPass>();
    AU.addRequired<OldSamplerCodeGen::OldSamplerCodeGenWrapperPass>();
  }

}; // end of struct Hello

char PlussWrapperPass::ID = 0;

static RegisterPass<PlussWrapperPass> X("pluss",
                                        "pluss locality analyzer",
                                        false /* Only looks at CFG */,
                                        true /* Analysis Pass */);

} // end of PlussWrapperPass namespace