//
//  ArrayInstrumentationUtils.hpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/20/21.
//

#ifndef ArrayInstrumentationUtils_hpp
#define ArrayInstrumentationUtils_hpp

#include <stdio.h>
#include "llvm/IR/Value.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/Regex.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;

Value * getArrayIndexExpr(Value *idxOp);
Value * getArrayNameExpr(Value *ptrOp);

#endif /* ArrayInstrumentationUtils_hpp */
