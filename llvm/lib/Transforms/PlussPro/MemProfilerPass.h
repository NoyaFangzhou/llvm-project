//
// Created by noya-fangzhou on 11/22/21.
//

#ifndef LLVM_MEMPROFILERPASS_H
#define LLVM_MEMPROFILERPASS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "PlussUtils.h"


struct InterestingMemoryAccess {
  Value *Addr = nullptr;
  bool IsWrite;
  bool IsInParallelRegion;
  unsigned Alignment;
  uint64_t TypeSize;
  Value *MaybeMask = nullptr;
};

namespace MemProfiler {

class MemProfiler {
  Function *F;
  LLVMContext *C;
  int LongSize;
  Type *IntptrTy;
public:
  MemProfiler(Function &Func);
  void instrumentMop(Instruction *I, const DataLayout &DL,
                     InterestingMemoryAccess &Access);
  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, uint32_t TypeSize, bool IsWrite,
                         bool IsInParallelRegion);
  void instrumentMaskedLoadOrStore(const DataLayout &DL, Value *Mask,
                                   Instruction *I, Value *Addr,
                                   unsigned Alignment, uint32_t TypeSize,
                                   bool IsWrite, bool IsInParallelRegion);
  void instrumentMemIntrinsic(MemIntrinsic *MI);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool instrumentFunction(Function &F);
  bool maybeInsertMemProfInitAtFunctionEntry(Function &F);
  bool insertDynamicShadowAtFunctionEntry(Function &F);
};


struct MemProfilerPass : FunctionPass {

  static char ID;
  MemProfilerPass();

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  MemProfiler *Profiler;
  /// If it is an interesting memory access, populate information
  /// about the access and return a InterestingMemoryAccess struct.
  /// Otherwise return None.
  Optional<InterestingMemoryAccess>
  isInterestingMemoryAccess(Instruction *I) const;
  bool maybeInsertMemProfInitAtFunctionEntry(Function &F);


};
}// end of namespace

#endif // LLVM_MEMPROFILERPASS_H
