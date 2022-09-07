//
//  ArrayInstrumentationUtils.cpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/20/21.
//

#include "ArrayInstrumentationUtils.hpp"

#define DEBUG_TYPE "array-instr-utils"

Value * getArrayIndexExpr(Value *idxOp) {
	if (isa<SExtInst>(idxOp)) {
		SExtInst *sextInst = dyn_cast<SExtInst>(idxOp);
		return getArrayIndexExpr(sextInst->User::getOperand(0));
	} else if (isa<BinaryOperator>(idxOp) || isa<LoadInst>(idxOp)) {
		return idxOp;
	}
	return NULL;
}

Value * getArrayNameExpr(Value *ptrOp) {
	if (!ptrOp->getName().str().empty()) {
		LLVM_DEBUG(dbgs() << ptrOp->getName() << "\n");
		return ptrOp;
	}
	LLVM_DEBUG(dbgs() << *ptrOp << "\n");
	if (isa<LoadInst>(ptrOp)) {
		LoadInst * ld = dyn_cast<LoadInst>(ptrOp);
		return getArrayNameExpr(ld->User::getOperand(0));
	} else if (isa<AllocaInst>(ptrOp)) {
		return ptrOp;
	}
	return NULL;
}
