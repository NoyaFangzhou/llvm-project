//
//  ArrayOMPInstrumentation.cpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/23/21.
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

#define DEBUG_TYPE "array-omp-instr"

using namespace llvm;
using namespace std;

namespace {
	struct ArrayOMPInstrumentation : public FunctionPass {
		static char ID;
		ArrayOMPInstrumentation() : FunctionPass(ID) {}
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
		
		int instrumentOMPGetTidFunc(Function &F, LLVMContext &Ctx, BasicBlock * B, GetElementPtrInst * I)
		{
			// Get an identifier for omp_get_thread_num()
			Type *retType = Type::getInt32Ty(Ctx);
			FunctionType *getTidFuncType = FunctionType::get(retType, false);
			FunctionCallee getTidFunc = F.getParent()->getOrInsertFunction("omp_get_thread_num", getTidFuncType);

			// Construct the struct and allocate space
//			Type* tidTy = Type::getInt32Ty(Ctx);
//			Instruction* tidAlloc = new AllocaInst(tidTy, 0, "tid", I);

			// Get a random number
			Instruction* getTidCall = CallInst::Create(getTidFunc, "", I);

			// Store the random number into the struct
//			Instruction* tidStore = new StoreInst(getTidCall, tidAlloc, I);
			
			
			std::vector<Type*> paramTypes = {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)};
			FunctionType *logFuncType = FunctionType::get(retType, paramTypes, false);
			FunctionCallee logFunc = F.getParent()->getOrInsertFunction("logompidx", logFuncType);
			
			// get the array index
			Value * arrayIdx = I->getOperand(1);
			// get the name of the array
			Value * arrayName = getArrayNameExpr(I->getPointerOperand());
			// right there, all irrelevant instruction can be removed
			// irrelevant instructions are those not in the def-use chain
			if (!arrayIdx || !arrayName)
				return -1;
			// Create the ArrayName obj
			ArrayName * name = new ArrayName(arrayName, &F);
			LLVM_DEBUG(dbgs() << "Instrument array " << * arrayName << " "
					   << *arrayName << "\n");
			IRBuilder<> builder(I);
			// ++ here means inserting after instruction I
			builder.SetInsertPoint(B, ++builder.GetInsertPoint());
			Value * args[] = {getTidCall, builder.getInt32(Generator.CreateArrayID(name)), arrayIdx};
			builder.CreateCall(logFunc.getFunctionType(), logFunc.getCallee(), args);
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
			IRBuilder<> builder(I);
			// ++ here means inserting after instruction I
			builder.SetInsertPoint(B, ++builder.GetInsertPoint());
			Value * args[] = {builder.getInt32(Generator.CreateArrayID(name)), arrayIdx};
			builder.CreateCall(func.getFunctionType(), func.getCallee(), args);
			return 0;
		}
		
		void getAnalysisUsage(AnalysisUsage & AU) const override {
			AU.setPreservesAll();
			AU.addRequired<FunctionInlineAnalysis::FunctionInlineAnalysis>();
		}

		/// Call for every function
		virtual bool runOnFunction(Function &F) override {
			int ret;
			DenseSet<StringRef> candidates = getAnalysis<FunctionInlineAnalysis::FunctionInlineAnalysis>().functions;
			bool beginInstr = candidates.find(F.getName()) != candidates.end();
			
			if (!beginInstr)
				return true;
			// Get the function to call from our runtime library.
			LLVMContext &Ctx = F.getContext();
			LLVM_DEBUG(dbgs() << "Function " << F.getName() << "\n");
			// Here we store all instructions that the array idx instruction depends
			unordered_set<Instruction *> defs;
			for (auto &B : F) {
				for (auto I = B.begin(); I != B.end(); ++I) {
					Instruction & inst = *I;
					if (isa<GetElementPtrInst>(&inst)) {
						GetElementPtrInst * arrayInst = dyn_cast<GetElementPtrInst>(&inst);
						LLVM_DEBUG(dbgs() << "Instrument after " << *arrayInst << "\n");
						ret = instrumentOMPGetTidFunc(F, Ctx, &B, arrayInst);
						if (ret < 0)
							LLVM_DEBUG(dbgs() << " FAILED \n");
						else
							LLVM_DEBUG(dbgs() << " SUCCESS \n");
					}
				}
			}
			return true;
		}
	};
}

char ArrayOMPInstrumentation::ID = 0;
static RegisterPass<ArrayOMPInstrumentation> X("array-omp-instr", "Instrument Array Access Pass in OpenMP region");

