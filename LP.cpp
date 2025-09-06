#include "KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h" 
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

enum Token{
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5,
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,
    tok_for = -9,
    tok_in = -10,
    tok_binary = -11,
    tok_unary = -12,
    tok_var = -13,
};
static std::string Identifier;
static double Num;

static int gettok() {
    static int LastChar = ' ';

    while(isspace(LastChar))
        LastChar = getchar();

    if(isalpha(LastChar)){
        Identifier = LastChar;
        while(isalnum((LastChar = getchar())))
            Identifier += LastChar;
        if(Identifier == "def")
            return tok_def;
        if(Identifier == "extern")
            return tok_extern;
        if(Identifier == "if")
            return tok_if;
        if(Identifier == "then")
            return tok_then;
        if(Identifier == "else")
            return tok_else;
        if(Identifier == "for")
            return tok_for;
        if(Identifier == "in")
            return tok_in;
        if(Identifier == "binary")
            return tok_binary;
        if(Identifier == "unary")
            return tok_unary;
        if(Identifier == "var")
            return tok_var;
        return tok_identifier;
    }

    if(isdigit(LastChar) || LastChar == '.'){
        std::string NumStr;
        int dec = 0;
        if(LastChar == '.'){
            dec++;
        }
        while(dec < 2 && isdigit(LastChar = getchar())){
            NumStr += LastChar;
        }
        Num = strtod(NumStr.c_str() , 0);
        return tok_number;
    }

    if(LastChar == '#'){
        do
            LastChar = getchar();
        while(LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if(LastChar != EOF)
            return gettok();
    }

    if(LastChar == EOF)
        return tok_eof;

    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}

namespace{
class ExprAST{
    public:
        virtual ~ExprAST() = default;
        virtual Value *codegen() = 0;
};

class NumberExprAST : public ExprAST{
    double Val;

    public:
        NumberExprAST(double Val) : Val(Val) {}
        Value *codegen() override;
};

class VariableExprAST : public ExprAST{
    std::string Name;

    public:
        VariableExprAST(std::string &Name) : Name(Name) {}
        Value *codegen() override;
};

class BinaryExprAST : public ExprAST{
    char Op;
    std::unique_ptr<ExprAST> LHS , RHS;

    public:
        BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, 
                                            std::unique_ptr<ExprAST> RHS)
                                        : Op(Op) , LHS(std::move(LHS)) , RHS(std::move(RHS)) {}
        Value *codegen() override;
};

class UnaryExprAST : public ExprAST{
    char Opcode;
    std::unique_ptr<ExprAST> Operand;

    public:
        UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
                            : Opcode(Opcode), Operand(std::move(Operand)) {}
        Value *codegen() override;
};  

class CallExprAST : public ExprAST{
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

    public:
        CallExprAST(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> Args)
                                                : Callee(Callee), Args(std::move(Args)) {}
        Value *codegen() override;
};

class PrototypeAST{
    std::string Name;
    std::vector<std::string> Args;
    bool isOperator;
    unsigned Precedence;    
    public:
        PrototypeAST(std::string &Name, std::vector<std::string> Args)
                            : Name(Name), Args(std::move(Args)) {}
        Function *codegen();
        const std::string &getName() const { return Name; }

        bool isUnaryOp() const { return isOperator && Args.size() == 1; }
        bool isBinaryOp() const{ return isOperator && Args.size() == 2; }

        char getOperatorName() const{
            assert(isUnaryOp() || isBinaryOp());
            return Name[Name.size() - 1];   
        }
        unsigned getBinaryPrecedence() const { return Precedence; }
};

class FunctionAST{
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Body)
                                        : Proto(std::move(Proto)) , Body(std::move(Body)) {}
        Function *codegen();
};

class IfExprAST : public ExprAST{
    std::unique_ptr<ExprAST> Cond, Then, Else;

    public:
        IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then, 
                            std::unique_ptr<ExprAST> Else) : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
        Value *codegen() override;
};

class ForExprAST : public ExprAST{
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;
    ForExprAST(std::string VarName, std::unique_ptr<ExprAST> Start, std::unique_ptr<ExprAST> End,
                    std::unique_ptr<ExprAST> Step, std::unique_ptr<ExprAST> Body)
                : VarName(VarName), Start(std::move(Start)), End(std::move(End)), Step(std::move(Step)), Body(std::move(Body)) {}

    Value *codegen() override;
};

class VarExprAST : public ExprAST{
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;

    public:
        VarExprAST(std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>>, std::unique_ptr<ExprAST> Body)
            : VarNames(std::move(VarNames)), Body(std::move(Body)){}

    Value *codegen() override;
};
}

static int CurTok;
static int getNextToken(){
    return CurTok = gettok();
}

static std::map<char, int> BinopPrecedence;

static int GetTokPrecendence(){
    if(!isascii(CurTok))
        return -1;
    
    int TokPrec = BinopPrecedence[CurTok];
    if(TokPrec <= 0)
        return -1;

    return TokPrec;
}

std::unique_ptr<ExprAST> LogError(const char* Str){
    fprintf(stderr, "Error: %s \n", Str);
    return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char* Str){
    LogError(Str);
    return nullptr;
}
static std::unique_ptr<ExprAST> ParseExpression();
static std::unique_ptr<ExprAST> ParseNumberExpr() {
    auto Result = std::make_unique<NumberExprAST>(Num);
    getNextToken();
    return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseParensExpr() {
    getNextToken();
    auto V = ParseExpression();
    if(!V)
        return nullptr;
    
    if(CurTok != ')')
        return LogError("Expected )");
    
    getNextToken();
    return V;
}

static std::unique_ptr<ExprAST> ParseFunctionCall() {
    std::string FunctionName = Identifier;

    getNextToken();

    if(CurTok != '(')
        return std::make_unique<VariableExprAST>(FunctionName);
    getNextToken();
    std::vector<std::unique_ptr<ExprAST>> Args;
    if(CurTok != ')' ){
        while(true){
            if(auto Arg = ParseExpression())
                Args.push_back(Arg);
            else   
                return nullptr;
            if(CurTok == ')')
                break;
            if(CurTok != ',')
                return LogError("Expected ) or , in argument list");
            getNextToken();
        }
    }
    
    getNextToken();

    return std::make_unique<CallExprAST>(FunctionName, std::move(Args));
}

static std::unique_ptr<ExprAST> ParseIfExpr(){
    getNextToken();

    auto Cond = ParseExpression();
    if(!Cond)
        return nullptr;
    
    if(CurTok !=tok_then)
        return LogError("Expected Then");
    getNextToken();
    auto Then = ParseExpression();
    if(!Then)
        return nullptr;
    if(CurTok != tok_else)
        return LogError("Expected Else");
    getNextToken();
    auto Else = ParseExpression();
    if(!Else)
        return nullptr;
    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then), std::move(Else));
}

static std::unique_ptr<ExprAST> ParseForExpr(){
    getNextToken();

    if(CurTok != tok_identifier)
        return LogError("Expected identifier after for");

    std::string IdName = Identifier;
    getNextToken();

    if(CurTok != '=')
        return LogError("Expected '=' after for");
    getNextToken();

    auto Start = ParseExpression();
    if(!Start)
        return nullptr;
    if(CurTok != ',')
        return LogError("Expected ',' after start value");
    getNextToken();

    auto End = ParseExpression();
    if(!End)
        return nullptr;
    
    std::unique_ptr<ExprAST> Step;
    if(CurTok == ','){
        getNextToken();
        Step = ParseExpression();
        if(!Step)
            return nullptr;
    }

    if(CurTok != tok_in)
        return LogError("expected in after for");
    getNextToken();

    auto Body = ParseExpression();
    if(!Body)
        return nullptr;

    return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End), std::move(Step), std::move(Body));
}

static std::unique_ptr<ExprAST> ParseVarExpr(){
    getNextToken();

    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    if(CurTok !=tok_identifier)
        return LogError("Expected identifier after var");
    while(true){
        std::string Name = Identifier;
        getNextToken();

        std::unique_ptr<ExprAST> Init = nullptr;
        if(CurTok == '='){
            getNextToken();

            Init = ParseExpression();
            if(!Init)
                return nullptr;
        }
        VarNames.push_back(std::make_pair(Name, std::move(Init)));

        if(CurTok != ',') 
            break;
        getNextToken();

        if(CurTok !=tok_identifier)
            return LogError("expected identifier list after var");
    }   
    if(CurTok != tok_in)
        return LogError("expected 'in' keyword after 'var' ");
    getNextToken();

    auto Body = ParseExpression();
    if(!Body)
        return nullptr;
    
    return std::make_unique<VarExprAST>(std::move(VarNames), std::move(Body));
}

static std::unique_ptr<ExprAST> ParsePrimary(){
    switch(CurTok){
        default:
            return LogError("unknown token when encountering an expression");
        case tok_identifier:
            return ParseFunctionCall();
        case tok_number:
            return ParseNumberExpr();
        case tok_if:
            return ParseIfExpr();
        case tok_for:
            return ParseForExpr();
        case tok_var:
            return ParseVarExpr();
        case '(':
            return ParseParensExpr();
    }
}

static std::unique_ptr<ExprAST> ParseUnaryExpr(){
    if(!isascii(CurTok) || CurTok != '(' || CurTok != ')')
        return ParsePrimary();
    int Opc = CurTok;
    getNextToken();
    if(auto Operand = ParseUnaryExpr())
        return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
    return nullptr; 
}

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS){
    while(true){
        int TokPrec = GetTokPrecendence();

        if(TokPrec < ExprPrec)
            return LHS;

        int BinOp = CurTok;
        getNextToken();

        auto RHS = ParseUnaryExpr();
        if(!RHS)
            return nullptr;
        int NextPrec = GetTokPrecendence();
        if(TokPrec < NextPrec){
            RHS = ParseBinOpRHS(TokPrec+1, std::move(RHS));
            if(!RHS)
                return nullptr;
        }
        LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
    }
}

static std::unique_ptr<ExprAST> ParseExpression(){
    auto LHS = ParseUnaryExpr();
    if(!LHS){
        return nullptr;
    }
    return ParseBinOpRHS(0, std::move(LHS));
}


static std::unique_ptr<PrototypeAST> ParsePrototype(){
    std::string FnName;
    unsigned Kind = 0;
    unsigned BinaryPrecedence = 3;
    switch(CurTok){
    default:
        return LogErrorP("Expected Function Name in Prototype");
    case tok_identifier:
        FnName = Identifier;
        Kind = 0;
        getNextToken();
        break;
    case tok_unary:
        getNextToken();
        if(!isascii(CurTok))
            return LogErrorP("Expected unary operator");
        FnName = "unary";
        FnName += char(CurTok);
        Kind = 1;
        getNextToken();
        break;
    case tok_binary:
        getNextToken();
        if(!isascii(CurTok))
            return LogErrorP("Expected Binary Operator");
        Kind = 2;
        FnName = "binary";
        FnName += (char)CurTok;
        getNextToken();
        if(CurTok == tok_number){
            if(Num < 1 || Num > 100){
                return LogErrorP("Invalid Precedence");
            }
            BinaryPrecedence = Num;
            getNextToken();
        }
        break;
    }
    if(CurTok != '(')
        return LogErrorP("Expected ( in prototype");
    std::vector<std::string> ArgNames;
    while(getNextToken() == tok_identifier)
        ArgNames.push_back(Identifier);
    if(CurTok != ')')
        return LogErrorP("Expected )");
    getNextToken();

    if(Kind && ArgNames.size() != Kind)
        return LogErrorP("Unexpected number of arguments");
    return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames), Kind!=0, BinaryPrecedence);
}

static std::unique_ptr<FunctionAST> ParseDefinition(){
    getNextToken();
    auto Proto = ParsePrototype();
    if(!Proto)
        return nullptr;
    if(auto E = ParseExpression())
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    return nullptr;
}

static std::unique_ptr<FunctionAST> ParseTopLevelExpr(){
    if (auto E = ParseExpression()) {
        // Make an anonymous proto.
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                    std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
      }
      return nullptr;
}

static std::unique_ptr<PrototypeAST> ParseExtern(){
    getNextToken();
    return ParsePrototype();
}

static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string,  AllocaInst*> NamedValues;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::unique_ptr<FunctionPassManager> TheFPM;
static std::unique_ptr<LoopAnalysisManager> TheLAM;
static std::unique_ptr<FunctionAnalysisManager> TheFAM;
static std::unique_ptr<CGSCCAnalysisManager> TheCGAM;
static std::unique_ptr<ModuleAnalysisManager> TheMAM;
static std::unique_ptr<PassInstrumentationCallbacks> ThePIC;
static std::unique_ptr<StandardInstrumentations> TheSI;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

static ExitOnError ExitOnErr;

Value *LogErrorV(const char *Str){
    LogError(Str);
    return nullptr;
}

Function *getFunction(std::string Name){
    if(auto *F = TheModule->getFunction(Name))
        return F;
    auto F1 = FunctionProtos.find(Name);
    if(F1 != FunctionProtos.end())
        return F1->second->codegen();

    return nullptr;
}

static AllocaInst* CreateEntryBlockAlloca(Function *TheFunction, StringRef VarName){
    IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type::getDoubleTy(*TheContext), nullptr, VarName);
}

Value *NumberExprAST::codegen(){
    return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *UnaryExprAST::codegen(){
    Value *OperandV = Operand->codegen();
    if(!OperandV)
        return nullptr;
    
    Function *F = getFunction(std::string("unary") + Opcode);
    if(!F)
        return LogErrorV("Unknown unary operator");
    return Builder->CreateCall(F, OperandV, "unop");
}

Value *BinaryExprAST::codegen(){
    if(Op == '='){
        VariableExprAST * LHSE = static_cast<VariableExprAST*>(LHS.get());
        if(!LHSE)
            return LogErrorV("destination of = must be a variable");
        Value *Val = RHS->codegen();
        if(!Val)
            return nullptr;
        Value *Variable = NamedValues[LHSE->getName()];
        if(!Variable)
            return LogErrorV("Unkown variable name");
        Builder->CreateStore(Val, Variable);
        return Val;
    }
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();

    if( !L || !R )
        return nullptr;
    switch(Op){
    case '+':
        return Builder->CreateFAdd(L,R,"add");
    case '-':
        return Builder->CreateFSub(L,R,"subs");
    case '*':
        return Builder->CreateFMul(L,R,"mul");
    case '/':
        return Builder->CreateFDiv(L,R,"div");
    case '<':
        L = Builder->CreateFCmpULT(L,R,"cmp");
        return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
    default:
        break;
    }
    Function *F = getFunction(std::string("binary") + Op);
    assert(F && "binary operator not found!");
    Value *Ops[] = {L, R};
    return Builder->CreateCall(F, Ops, "binop");
}

Value *CallExprAST::codegen(){
    Function *Calleef = getFunction(Callee);
    if(!Calleef)
        return LogErrorV("unkown function name");
    if(Calleef->arg_size() != Args.size())
        return LogErrorV("invalid # of arguments");
    std::vector<Value*>ArgsV;
    for(unsigned i = 0, e=Args.size(); i != e; ++i){
        ArgsV.push_back(Args[i]->codegen());
        if(!ArgsV.back())
            return nullptr;
    }
    return Builder->CreateCall(Calleef,ArgsV, "call");
}

Value *IfExprAST::codegen(){
    Value *CondV = Cond->codegen();
    if(!CondV)
        return nullptr;
    CondV = Builder->CreateFCmpONE(CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");
    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    
    BasicBlock *ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
    BasicBlock *ElseBB = BasicBlock::Create(*TheContext, "else");
    BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");
    Builder->CreateCondBr(CondV, ThenBB, ElseBB);
    Builder->SetInsertPoint(ThenBB);

    Value* ThenV = Then->codegen();
    if(!ThenV)
        return nullptr;
    Builder->CreateBr(MergeBB);
    ThenBB = Builder->GetInsertBlock();

    TheFunction->insert(TheFunction->end(), ElseBB);
    Builder->SetInsertPoint(ElseBB);
    Value *ElseV = Else->codegen();
    if(!ElseV)
        return nullptr;
    Builder->CreateBr(MergeBB);
    ElseBB = Builder->GetInsertBlock();

    TheFunction->insert(TheFunction->end(), MergeBB);
    Builder->SetInsertPoint(MergeBB);
    PHINode *PN = Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp");

    PN->addIncoming(ThenV, ThenBB);
    PN->addIncoming(ElseV, ElseBB);
    return PN;
}

Value *ForExprAST::codegen(){
    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    
    Value *StartV = Start->codegen();
    if(!StartV)
        return nullptr;

    Builder->CreateStore(StartV, Alloca);
    BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop", TheFunction);
    Builder->CreateBr(LoopBB);

    Builder->SetInsertPoint(LoopBB);

    AllocaInst *OldVal = NamedValues[VarName];
    NamedValues[VarName] = Alloca;

    if(!(Body->codegen()))
        return nullptr;

    Value *StepVal = nullptr;
    if(Step){
        StepVal = Step->codegen();
        if(!StepVal)
            return nullptr;
    }
    else{
        StepVal = ConstantFP::get(*TheContext, APFloat(1.0));
    }
    Value *EndCond = End->codegen();
    if(!EndCond)
        return nullptr;
    
    Value *CurVar = Builder->CreateLoad(Alloca->getAllocatedType(), Alloca, VarName.c_str());
    
    Value *NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar");
    Builder->CreateStore(NextVar, Alloca);


    EndCond = Builder->CreateFCmpONE(EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");
    BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "afterbb");
    Builder->CreateCondBr(EndCond, LoopBB, AfterBB);
    Builder->SetInsertPoint(AfterBB);

    if(OldVal)
        NamedValues[VarName] = OldVal;
    else   
        NamedValues.erase(VarName);
    return Constant::getNullValue(Type::getDoubleTy(*TheContext));

}

Value *VarExprAST::codegen(){
    std::vector<AllocaInst *> OldBindings;

    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    for(unsigned i = 0,e = VarNames.size(); i != e; ++i){
        const std::string &VarName = VarNames[i].first;
        ExprAST *Init = VarNames[i].second.get();

        Value *InitVal;
        if (Init) {
            InitVal = Init->codegen();
            if (!InitVal)
                return nullptr;
        } 
        else {
            InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
        }
        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
        Builder->CreateStore(InitVal, Alloca);
        OldBindings.push_back(NamedValues[VarName]);
        NamedValues[VarName] = Alloca;
    }
    Value *BodyVal = Body->codegen();
    if(!BodyVal)
        return nullptr;
    for (unsigned i = 0; e = VarNames.size(); i!=e; ++i)
        NamedValues[VarNames[i].first] = OldBindings[i];
    return BodyVal;
}

Function *PrototypeAST::codegen(){
    std::vector<Type*> Doubles(Args.size(),
                                        Type::getDoubleTy(*TheContext));
    FunctionType *Ft = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
    Function *F = Function::Create(Ft, Function::ExternalLinkage, Name, TheModule.get());
    unsigned Idx = 0;
    for(auto &Arg : F->args())
        Arg.setName(Args[Idx++]);
    return F;                                  
}

Function *FunctionAST::codegen(){
    auto &P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    Function *TheFunction = getFunction(P.getName());   
    
    if(!TheFunction)
        return nullptr;
    
    if(P.isBinaryOp())
        BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();
    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    NamedValues.clear();
    for(auto &Arg : TheFunction->args()){
        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
        Builder->CreateStore(&Arg, Alloca);
        NamedValues[std::string(Arg.getName())] = Alloca;
    }

    if(Value *RetV = Body->codegen()){
        Builder->CreateRet(RetV);

        verifyFunction(*TheFunction);

        TheFPM->run(*TheFunction, *TheFAM);
        return TheFunction;
    }
    TheFunction->eraseFromParent();

    if(P.isBinaryOp)
        BinopPrecedence.erase(P.getOperatorName());
    return nullptr;
}

static void InitializeModuleandManagers(void){
    TheContext = std::make_unique<LLVMContext>();
    TheModule = std::make_unique<Module>("KaleidoscopeJIT", *TheContext);
    TheModule->setDataLayout(TheJIT->getDataLayout());

    Builder = std::make_unique<IRBuilder<>>(*TheContext);

    TheFPM = std::make_unique<FunctionPassManager>();
    TheLAM = std::make_unique<LoopAnalysisManager>();
    TheFAM = std::make_unique<FunctionAnalysisManager>();
    TheCGAM = std::make_unique<CGSCCAnalysisManager>();
    TheMAM = std::make_unique<ModuleAnalysisManager>();
    ThePIC = std::make_unique<PassInstrumentationCallbacks>();
    TheSI = std::make_unique<StandardInstrumentations>(*TheContext , true);

    TheSI->registerCallbacks(*ThePIC, TheMAM.get());

    TheFPM->addPass(PromotePass());
    TheFPM->addPass(InstCombinePass());
    TheFPM->addPass(ReassociatePass());
    TheFPM->addPass(GVNPass());
    TheFPM->addPass(SimplifyCFGPass());

    PassBuilder PB;
    PB.registerModuleAnalyses(*TheMAM);
    PB.registerFunctionAnalyses(*TheFAM);
    PB.crossRegisterProxies(*TheLAM, *TheFAM, *TheCGAM, *TheMAM);
}

static void HandleDefinition(){
    if(auto FnAST = ParseDefinition()){
        if(auto *FnIR = FnAST->codegen()){
            fprintf(stderr, "Read function definition:");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            ExitOnErr(TheJIT->addModule(ThreadSafeModule(std::move(TheModule), std::move(TheContext))));
            InitializeModuleandManagers();
        }
    }
    else{
        getNextToken();
    }
}

static void HandleExtern(){
    if(auto ProtoAST = ParseExtern()){
        if(auto *ProtoIR = ProtoAST->codegen()){
            fprintf(stderr, "Read extern: ");
            ProtoIR->print(errs());
            fprintf(stderr, "\n");
            FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
    }
    else{
        getNextToken();
    }
}

static void HandleTopLevelExpr(){
    if(auto FnAST = ParseTopLevelExpr()){
        if(FnAST->codegen()){
            auto RT = TheJIT->getMainJITDylib().createResourceTracker();
            
            auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));

            ExitOnErr(TheJIT->addModule(std::move(TSM), RT));

            InitializeModuleandManagers();
            auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
            assert(ExprSymbol && "Function not found");

            double (*FP)() = ExprSymbol.getAddress.toPtr<double(*) ()> ();
            fprintf(stderr, "Evaluated to %f\n", FP());

            ExitOnErr(RT->remove());
        }
    }
    else{
        getNextToken();
    }
}

static void Mainloop(){
    while(true){
        fprintf(stderr, "ready> ");
        switch(CurTok){
            case tok_eof:
                return;
            case ';':
                getNextToken();
                break;
            case tok_def:
                HandleDefinition();
                break;
            case tok_extern:
                HandleExtern();
                break;
            case tok_if:
                
            default:
                HandleTopLevelExpr();
                break;
        }
    }
}

static std::unique_ptr<KaleidoscopeJIT> TheJIT;

int main(){
    InitializeNativeTarget();
    InitializeNativeTargetASMPrinter();
    InitializeNativeTargetASMParser();
    
    BinopPrecedence['/'] = 5;
    BinopPrecedence['*'] = 4;
    BinopPrecedence['+'] = 3;
    BinopPrecedence['-'] = 3;
    BinopPrecedence['<'] = 2;   
    
    fprintf(stderr, "code>");
    getNextToken();

    TheJIT = std::make_unique<KaleidoscopeJIT>();

    MainLoop();

    return 0;
}