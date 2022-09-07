//
//  ArrayIDGenerator.hpp
//  ArrayInstrumentationPass
//
//  Created by noya-fangzhou on 4/19/21.
//

#ifndef ArrayIDGenerator_hpp
#define ArrayIDGenerator_hpp

#include <stdio.h>
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Function.h"
#include <unordered_map>

using namespace std;
using namespace llvm;


/// To avoid dumplicate name appears in different function
/// We put the name and its context together
class ArrayName {
public:
	Value * name;
	Function * func;
	ArrayName();
	ArrayName(Value * name, Function * F);
	void print();
};


class ArrayIDGenerator {
	int ID = 0;
	unordered_map<int, ArrayName *> ArrayIDTable;
	
public:
	ArrayIDGenerator();
	
	int CreateArrayID(ArrayName * array);
	
	int LookupID(ArrayName * array);
	
	ArrayName * LookupArray(int id);
	
	void print();
	
};

#endif /* ArrayIDGenerator_hpp */
