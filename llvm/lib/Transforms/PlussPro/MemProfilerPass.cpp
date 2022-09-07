//
// Created by noya-fangzhou on 11/22/21.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "MemProfilerPass.h"

#define DEBUG_TYPE "memprof"

constexpr char PlussMemProfTimerStartName[] = "pluss_timer_start";
constexpr char PlussMemProfTimerStopName[] = "pluss_timer_stop";
constexpr char PlussMemProfTimerPrintName[] = "pluss_timer_print";
constexpr char PlussMemProfInitName[] = "pluss_init";
constexpr char PlussMemProfFuncName[] = "pluss_access";
constexpr char PlussParallelMemProfFuncName[] = "pluss_parallel_access";
constexpr char PlussAETFuncName[] = "pluss_AET";
constexpr char PlussMemProfHistogramDumpName[] = "pluss_print_histogram";
constexpr char PlussMemProfMRCDumpName[] = "pluss_print_mrc";

cl::opt<string> FunctionToProf(
    "func-to-profile", cl::init(string()), cl::NotHidden,
    cl::desc("The function name to analyze"));

STATISTIC(NumInstructionsChecks, "Number of instructions being examined");
STATISTIC(NumInterestingInst, "Number of instructions to be profiled");
STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumSkippedStackReads, "Number of non-instrumented stack reads");
STATISTIC(NumSkippedStackWrites, "Number of non-instrumented stack writes");

namespace MemProfiler {

MemProfiler::MemProfiler(Function &Func)
{
  F = &Func;
  C = &(F->getParent()->getContext());
  LongSize = F->getParent()->getDataLayout().getPointerSizeInBits();
  IntptrTy = Type::getIntNTy(*C, LongSize);
}

void MemProfiler::instrumentMaskedLoadOrStore(const DataLayout &DL, Value *Mask,
                                              Instruction *I, Value *Addr,
                                              unsigned Alignment,
                                              uint32_t TypeSize, bool IsWrite,
                                              bool IsInParallelRegion) {
  auto *VTy = cast<FixedVectorType>(
      cast<PointerType>(Addr->getType())->getElementType());
  uint64_t ElemTypeSize = DL.getTypeStoreSizeInBits(VTy->getScalarType());
  unsigned Num = VTy->getNumElements();
  auto *Zero = ConstantInt::get(IntptrTy, 0);
  for (unsigned Idx = 0; Idx < Num; ++Idx) {
    Value *InstrumentedAddress = nullptr;
    Instruction *InsertBefore = I;
    if (auto *Vector = dyn_cast<ConstantVector>(Mask)) {
      // dyn_cast as we might get UndefValue
      if (auto *Masked = dyn_cast<ConstantInt>(Vector->getOperand(Idx))) {
        if (Masked->isZero())
          // Mask is constant false, so no instrumentation needed.
          continue;
        // If we have a true or undef value, fall through to instrumentAddress.
        // with InsertBefore == I
      }
    } else {
      IRBuilder<> IRB(I);
      Value *MaskElem = IRB.CreateExtractElement(Mask, Idx);
      Instruction *ThenTerm = SplitBlockAndInsertIfThen(MaskElem, I, false);
      InsertBefore = ThenTerm;
    }

    IRBuilder<> IRB(InsertBefore);
    InstrumentedAddress =
        IRB.CreateGEP(VTy, Addr, {Zero, ConstantInt::get(IntptrTy, Idx)});
    instrumentAddress(I, InsertBefore, InstrumentedAddress, ElemTypeSize,
                      IsWrite, IsInParallelRegion);
  }
}

void MemProfiler::instrumentMop(Instruction *I, const DataLayout &DL,
                                InterestingMemoryAccess &Access) {
  // Skip instrumentation of stack accesses unless requested.
  // getUnderlyingObject() is available after LLVM 13
#if 0
  if (isa<AllocaInst>(getUnderlyingObject(Access.Addr))) {
    if (Access.IsWrite)
      ++NumSkippedStackWrites;
    else
      ++NumSkippedStackReads;
    return;
  }
#endif

  if (Access.IsWrite)
    NumInstrumentedWrites++;
  else
    NumInstrumentedReads++;

  if (Access.MaybeMask) {
    instrumentMaskedLoadOrStore(DL, Access.MaybeMask, I, Access.Addr,
                                Access.Alignment, Access.TypeSize,
                                Access.IsWrite, Access.IsInParallelRegion);
  } else {
    // Since the access counts will be accumulated across the entire allocation,
    // we only update the shadow access count for the first location and thus
    // don't need to worry about alignment and type size.
    instrumentAddress(I, I, Access.Addr, Access.TypeSize, Access.IsWrite,
                      Access.IsInParallelRegion);
  }
}

void MemProfiler::instrumentAddress(Instruction *OrigIns,
                                    Instruction *InsertBefore, Value *Addr,
                                    uint32_t TypeSize, bool IsWrite,
                                    bool IsInParallelRegion)
{
  IRBuilder<> IRB(InsertBefore);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);

  SmallVector<Type *, 2> paramTypes{1, IntptrTy};
  Type *retType = Type::getVoidTy(*C);
  FunctionType *logFuncType = FunctionType::get(retType,
                                                paramTypes, false);
  if (IsInParallelRegion) {
    FunctionCallee logFunc =
        F->getParent()->getOrInsertFunction(PlussParallelMemProfFuncName, logFuncType);
    IRB.CreateCall(logFunc, AddrLong);
  } else {
    FunctionCallee logFunc =
        F->getParent()->getOrInsertFunction(PlussMemProfFuncName, logFuncType);
    IRB.CreateCall(logFunc, AddrLong);
  }

#if 0
  // Create an inline sequence to compute shadow location, and increment the
  // value by one.
  Type *ShadowTy = Type::getInt64Ty(*C);
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  Value *ShadowAddr = IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy);
  Value *ShadowValue = IRB.CreateLoad(ShadowTy, ShadowAddr);
  Value *Inc = ConstantInt::get(Type::getInt64Ty(*C), 1);
  ShadowValue = IRB.CreateAdd(ShadowValue, Inc);
  IRB.CreateStore(ShadowValue, ShadowAddr);
#endif
}


char MemProfilerPass::ID = 0;
static RegisterPass<MemProfilerPass>
    X("memprof", "Pass that instruments all heap memory accesses");

MemProfilerPass::MemProfilerPass() : FunctionPass(ID) {}

Optional<InterestingMemoryAccess>
MemProfilerPass::isInterestingMemoryAccess(Instruction *I) const {
#if 0
  // Do not instrument the load fetching the dynamic shadow address.
  if (DynamicShadowOffset == I)
    return None;
#endif
  LLVM_DEBUG(dbgs() << "Check instruction " << *I << "\n");
  InterestingMemoryAccess Access;

  const DataLayout &DL = I->getModule()->getDataLayout();
  bool IsInParallelRegion =
      (I->getFunction()->getName().str().find("omp_outlined") != string::npos);
  if (IsInParallelRegion)
    LLVM_DEBUG(dbgs() << *I << " is inside OMP region\n");
  Access.IsInParallelRegion = IsInParallelRegion;
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
//    if (!ClInstrumentReads)
//      return None;
    Access.IsWrite = false;
    Access.TypeSize = DL.getTypeStoreSizeInBits(LI->getType());
    Access.Alignment = LI->getAlignment();
    Access.Addr = LI->getPointerOperand();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
//    if (!ClInstrumentWrites)
//      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(SI->getValueOperand()->getType());
    Access.Alignment = SI->getAlignment();
    Access.Addr = SI->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
//    if (!ClInstrumentAtomics)
//      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(RMW->getValOperand()->getType());
    Access.Alignment = 0;
    Access.Addr = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
//    if (!ClInstrumentAtomics)
//      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(XCHG->getCompareOperand()->getType());
    Access.Alignment = 0;
    Access.Addr = XCHG->getPointerOperand();
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    auto *F = CI->getCalledFunction();
    if (F && (F->getIntrinsicID() == Intrinsic::masked_load ||
              F->getIntrinsicID() == Intrinsic::masked_store)) {
      unsigned OpOffset = 0;
      if (F->getIntrinsicID() == Intrinsic::masked_store) {
//        if (!ClInstrumentWrites)
//          return None;
        // Masked store has an initial operand for the value.
        OpOffset = 1;
        Access.IsWrite = true;
      } else {
//        if (!ClInstrumentReads)
//          return None;
        Access.IsWrite = false;
      }

      auto *BasePtr = CI->getOperand(0 + OpOffset);
      auto *Ty = cast<PointerType>(BasePtr->getType())->getElementType();
      Access.TypeSize = DL.getTypeStoreSizeInBits(Ty);
      if (auto *AlignmentConstant =
          dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Access.Alignment = (unsigned)AlignmentConstant->getZExtValue();
      else
        Access.Alignment = 1; // No alignment guarantees. We probably got Undef
      Access.MaybeMask = CI->getOperand(2 + OpOffset);
      Access.Addr = BasePtr;
    }
  }

  if (!Access.Addr)
    return None;

  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  Type *PtrTy = cast<PointerType>(Access.Addr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() != 0)
    return None;

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Access.Addr->isSwiftError())
    return None;

  return Access;
}

bool MemProfilerPass::maybeInsertMemProfInitAtFunctionEntry(Function &F) {
  // For each NSObject descendant having a +load method, this method is invoked
  // by the ObjC runtime before any of the static constructors is called.
  // Therefore we need to instrument such methods with a call to __memprof_init
  // at the beginning in order to initialize our runtime before any access to
  // the shadow memory.
  // We cannot just ignore these methods, because they may call other
  // instrumented functions.

  FunctionType *initFuncType = FunctionType::get(Type::getVoidTy(
                                                     F.getParent()->getContext()),
                                                 false);
  IRBuilder<> IRB(&F.front(), F.front().begin());
  FunctionCallee initFunc = F.getParent()->getOrInsertFunction(PlussMemProfInitName, initFuncType);
  IRB.CreateCall(initFunc, {});

  FunctionCallee TimerStartFunc = F.getParent()->getOrInsertFunction(PlussMemProfTimerStartName, initFuncType);
  IRB.CreateCall(TimerStartFunc, {});

  return true;
}

bool MemProfilerPass::runOnFunction(Function &F)
{
  LLVM_DEBUG(
      dbgs() << "CHECK FUNCTION " << F.getName() << "\n";
      );
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
    return false;
  if (F.getName().startswith("pluss"))
    return false;
  if ((!FunctionToProf.empty() && F.getName().str().find(FunctionToProf) == string::npos)
      && (F.getName().str().find(".omp_outlined") == string::npos))
    return false;

  bool FunctionModified = false;
  // If needed, insert __memprof_init.
  // This function needs to be called even if the function body is not
  // instrumented.
  if (F.getName().str().find(FunctionToProf) != string::npos) {
    if (maybeInsertMemProfInitAtFunctionEntry(F)) {
      FunctionModified = true;
    }
  }

  LLVM_DEBUG(dbgs() << "MEMPROF instrumenting:\n" << F.getName() << "\n");

  Profiler = new MemProfiler(F);
  vector<Instruction *> InterestingMemOpToProfile;
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    NumInstructionsChecks++;
    if (isInterestingMemoryAccess(&*I) || isa<MemIntrinsic>(&*I))
      InterestingMemOpToProfile.push_back(&*I);
  }
  NumInterestingInst += InterestingMemOpToProfile.size();
  FunctionModified = !InterestingMemOpToProfile.empty();
  for (auto *Inst : InterestingMemOpToProfile) {
    Optional<InterestingMemoryAccess> Access =
          isInterestingMemoryAccess(Inst);
    if (Access)
      Profiler->instrumentMop(Inst, F.getParent()->getDataLayout(), *Access);
    else
      Profiler->instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
  }

  LLVM_DEBUG(dbgs() << "MEMPROF done instrumenting: " << FunctionModified << " "
                    << F.getName() << "\n");

  if (F.getName().str().find(FunctionToProf) != string::npos) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (isa<ReturnInst>(&*I)) {
        FunctionType *dumpFuncType =
            FunctionType::get(Type::getVoidTy(F.getParent()->getContext()), false);
        //  IRBuilder<> IRB(&F.back(), F.back().rbegin());

        IRBuilder<> IRB(&*I);
        IRB.SetInsertPoint(I->getParent(), IRB.GetInsertPoint());
        FunctionCallee AETFunc =
            F.getParent()->getOrInsertFunction(PlussAETFuncName, dumpFuncType);
        FunctionCallee TimerStopFunc = F.getParent()->getOrInsertFunction(
            PlussMemProfTimerStopName, dumpFuncType);
        FunctionCallee TimerPrintFunc = F.getParent()->getOrInsertFunction(
            PlussMemProfTimerPrintName, dumpFuncType);
        FunctionCallee HistogramDumpFunc = F.getParent()->getOrInsertFunction(
            PlussMemProfHistogramDumpName, dumpFuncType);
        FunctionCallee MRCDumpFunc = F.getParent()->getOrInsertFunction(
            PlussMemProfMRCDumpName, dumpFuncType);
        IRB.CreateCall(AETFunc, {});
        IRB.CreateCall(TimerStopFunc, {});
        IRB.CreateCall(TimerPrintFunc, {});
        IRB.CreateCall(HistogramDumpFunc, {});
        IRB.CreateCall(MRCDumpFunc, {});
      }
    }
  }

  return FunctionModified;
}

void MemProfilerPass::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.setPreservesAll();
}


} // end of namespace
