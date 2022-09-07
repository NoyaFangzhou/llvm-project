//
//  ArrayInstrumentationTest.cpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 5/2/21.
//

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "FunctionInliningAnalysis.hpp"
#include "ArrayIDGenerator.hpp"
#include "ArrayInstrumentationUtils.hpp"

#include <unordered_set>
#include <set>
#define DEBUG_TYPE "array-instr-test"

using namespace llvm;
using namespace std;

namespace {
	struct ArrayInstrumentationTest : public FunctionPass {
		static char ID;
		ArrayInstrumentationTest() : FunctionPass(ID) {}
		ArrayIDGenerator Generator;
		
		void findAllDefs(Value * v, unordered_set<Instruction *> &defs)
		{
			if (!isa<Instruction>(v)) return;
			Instruction * I = dyn_cast<Instruction>(v);
			defs.insert(I);
			for (Use &U : I->operands()) {
				Value * def = U.get();
				findAllDefs(def, defs);
			}
		}
		
		
		void findAllUses(Instruction * I, unordered_set<Instruction *> &users)
		{
			if (I->getNumUses() > 0) {
				for (User * U : I->users()) {
					if (Instruction *userInst = dyn_cast<Instruction>(U)) {
						users.insert(userInst);
						 findAllUses(userInst, users);
					}
				}
			}
		}
		
		
		int instrumentAnalysisFunc(FunctionCallee func, BasicBlock * B, Instruction * I)
		{
			IRBuilder<> builder(I);
			builder.SetInsertPoint(B, builder.GetInsertPoint());
			builder.CreateCall(func.getFunctionType(), func.getCallee());
			return 0;
		}
		
		/// Instrument a runtime function
		int instrumentMemoryAccess(Function * F, FunctionCallee func, BasicBlock * B, GetElementPtrInst * I)
		{
			// get the array index
			Value * arrayIdx = I->getOperand(1);
			// get the name of the array
			Value * arrayName = getArrayNameExpr(I->getPointerOperand());
			// right there, all irrelevant instruction can be removed
			// irrelevant instructions are those not in the def-use chain
			if (!arrayIdx || !arrayName)
				return -1;
			// Create the ArrayName obj
			ArrayName * name = new ArrayName(arrayName, F);
			LLVM_DEBUG(dbgs() << "Instrument array " << * arrayName << " "
					   << *arrayName << "\n");
			LLVM_DEBUG(dbgs() << "Array " << * arrayName << " has type "
					   << *(arrayName->getType()) << "\n");
			
			IRBuilder<> builder(I);
			builder.SetInsertPoint(B, ++builder.GetInsertPoint());
//			Value * args[] = {arrayName, I};
//			builder.CreateCall(func.getFunctionType(), func.getCallee(), args);
//							builder.CreateCall(func.getFunctionType(), func.getCallee(), args);
			Value * arrayBase = builder.CreatePtrToInt(arrayName, Type::getInt64Ty(F->getContext()), "arraybase");
			Value * arrayAddr = builder.CreatePtrToInt(I, Type::getInt64Ty(F->getContext()), "arrayaddr");
			
			
			if (isa<PtrToIntInst>(arrayBase) && isa<PtrToIntInst>(arrayAddr)) {
				PtrToIntInst * arrayAddrInst = dyn_cast<PtrToIntInst>(arrayAddr);
				LLVM_DEBUG(dbgs() << "Instrument logaddr() after " << * arrayBase << " and "
						   << *arrayAddr << "\n");
				IRBuilder<> builder(arrayAddrInst);
				// ++ here means inserting after instruction I
				builder.SetInsertPoint(B, ++builder.GetInsertPoint());
				Value * args[] = {arrayBase, arrayAddr};
				builder.CreateCall(func.getFunctionType(), func.getCallee(), args);
			}
			return 0;
		}
		
		void getAnalysisUsage(AnalysisUsage & AU) const override {
			AU.setPreservesAll();
			AU.addRequired<FunctionInlineAnalysis::FunctionInlineAnalysis>();
		}

		/// Call for every function
		virtual bool runOnFunction(Function &F) override {
			
			int ret;
			int removeInstCnt = 0;
			
			DenseSet<StringRef> candidates = getAnalysis<FunctionInlineAnalysis::FunctionInlineAnalysis>().functions;
			// Get the function to call from our runtime library.
			LLVMContext &Ctx = F.getContext();
//			std::vector<Type*> paramTypes = {Type::getDoublePtrTy(Ctx), Type::getDoublePtrTy(Ctx)};
			std::vector<Type*> paramTypes = {Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx)};
			Type *retType = Type::getVoidTy(Ctx);
			FunctionType *logFuncType = FunctionType::get(retType, paramTypes, false);
			FunctionCallee logFunc = F.getParent()->getOrInsertFunction("logaddr", logFuncType);
			
			bool ismain = F.getName().contains("main");
			
			bool beginInstr = candidates.find(F.getName()) != candidates.end();
			LLVM_DEBUG(dbgs() << "Function " << F.getName() << " " << beginInstr << "\n");
			if (beginInstr || ismain) {
				// Here we store all instructions that the array idx instruction depends
				unordered_set<Instruction *> defs;
				for (auto &B : F) {
					for (auto I = B.begin(); I != B.end(); ++I) {
						Instruction & inst = *I;
						if (beginInstr && isa<GetElementPtrInst>(&inst)) {
							GetElementPtrInst * arrayInst = dyn_cast<GetElementPtrInst>(&inst);
							LLVM_DEBUG(dbgs() << "Instrument after " << *arrayInst << "\n");
							LLVM_DEBUG(dbgs() << "Array base pointer is in type " << *((arrayInst->getPointerOperand())->getType()) << "\n");
//							ret = instrumentMemoryAccess(&F, logFunc, &B, arrayInst);
//							if (ret < 0)
//								LLVM_DEBUG(dbgs() << " FAILED \n");
//							else
//								LLVM_DEBUG(dbgs() << " SUCCESS \n");
						}
					}
				}
				if (ismain) {
					Generator.print();
				}
			}
			return true;
		}
	};
}

char ArrayInstrumentationTest::ID = 0;
static RegisterPass<ArrayInstrumentationTest> X("test", "Instrument Array Access Test Pass");
