//
//  ArrayIDGenerator.cpp
//  ArrayInstrumentationPass
//
//  Created by noya-fangzhou on 4/19/21.
//

#include "ArrayIDGenerator.hpp"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE 	"array-id-generator"


ArrayName::ArrayName() {}
ArrayName::ArrayName(Value * name, Function * F)
{
	this->name = name;
	this->func = F;
}

void ArrayName::print()
{
	errs() << "Array " << *(this->name) << " in Func " << this->func->getName();
}

ArrayIDGenerator::ArrayIDGenerator()
{
	this->ID = 0;
}

int ArrayIDGenerator::CreateArrayID(ArrayName * array)
{
	int ret = LookupID(array);
	if (ret > 0)
		return ret;
	ID++;
	ArrayIDTable.emplace(ID, array);
	return ID;
}

int ArrayIDGenerator::LookupID(ArrayName * array)
{
	unordered_map<int, ArrayName *>::iterator it = ArrayIDTable.begin();
	for (; it != ArrayIDTable.end(); ++it) {
		if (it->second->name == array->name && it->second->func == array->func) {
			return it->first;
		}
	}
	return -1;
}

ArrayName * ArrayIDGenerator::LookupArray(int id)
{
	unordered_map<int, ArrayName *>::iterator it = ArrayIDTable.find(id);
	if (it != ArrayIDTable.end()) {
		return it->second;
	}
	return NULL;
}

void ArrayIDGenerator::print()
{
	unordered_map<int, ArrayName *>::iterator it = ArrayIDTable.begin();
	for (; it != ArrayIDTable.end(); ++it) {
		it->second->print();
		errs() << "has ID " << it->first << "\n";
	}
}
