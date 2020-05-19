#include <boost/python.hpp>
#include<iostream>
#include <boost/python/module.hpp> 
#include <boost/python/def.hpp>
#include<string.h>
#include<vector>

#include "toy/Dialect.h"
#include "toy/Passes.h"


#include "mlir/Analysis/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <memory>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using namespace mlir::relay;
namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir
// __attribute__ ((__constructor__)) 
//   void pre_func(void) {
//         std::cout<<"pre_func\n";
//         mlir::registerDialect<mlir::relay::RelayDialect>();
// }
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
  Location(char* str,int a,int b)
  {
      file = std::make_shared<std::string>(str);
      line = a;
      col  = b;
  }
};

  mlir::Block* entryBlock;
  mlir::FuncOp function;
  static bool _register_dialects = [] {
  // ... call into what you want here
  mlir::registerDialect<mlir::relay::RelayDialect>();
  mlir::registerAllDialects();
  return true;
}();
static mlir::MLIRContext context;
class RelayImpl{
public:
    
    RelayImpl(mlir::MLIRContext &context) : builder(&context) {}
    mlir::Value var(const char* name, boost::python::list data,boost::python::list shape);
    mlir::Value transpose(mlir::Value operand);
    mlir::Value add(mlir::Value lhs,mlir::Value rhs);
    mlir::Value multiply(mlir::Value lhs,mlir::Value rhs);
    // mlir::Value batch_norm(mlir::Value data, mlir::Value gamma,mlir::Value beta, mlir::Value moving_mean,mlir::Value moving_var);
    // mlir::Value relu(mlir::Value operand);
    //mlir::Value conv2d(mlir::Value data,mlir::Value filter,int strides_h,int strides_w,int padding);
    mlir::Value conv2d(mlir::Value data,mlir::Value filter);
    // mlir::Value max_pool2d(mlir::Value operand);
    // mlir::Value global_avg_pool2d(mlir::Value operand);
    // mlir::Value batch_flatten(mlir::Value operand);
    // mlir::Value softmax(mlir::Value operand);
    // mlir::Value dense(mlir::Value data,mlir::Value weight);
    // mlir::Value bias_add(mlir::Value operand);
    mlir::FuncOp Function(boost::python::list free_vars,mlir::Value res);
    mlir::ModuleOp Module(mlir::FuncOp func);
    mlir::FuncOp Prototype(int number);
    void print(mlir::Value operand);
    void dumpMLIR(mlir::ModuleOp module);
    //void dumpMLIRAffine(mlir::ModuleOp module);
    // void dumpMLIRLLVM(mlir::ModuleOp module);
    // void dumpLLVM(mlir::ModuleOp module);
private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
    }
    mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
    }
    
};
RelayImpl getclass()
{
    //mlir::MLIRContext context;
    RelayImpl relay1(context);
    return relay1;
}
void dumpMLIRAffine(mlir::ModuleOp module);
void inferShape(mlir::ModuleOp module);
void dumpMLIRLLVM(mlir::ModuleOp module);
void dumpLLVM(mlir::ModuleOp module);
BOOST_PYTHON_MODULE(librelay) 
{ 
    using namespace boost::python;
    
 class_<RelayImpl>("RelayImpl",init<mlir::MLIRContext & >())
  .def("var", &RelayImpl::var)
  .def("transpose", &RelayImpl::transpose)
  .def("add", &RelayImpl::add)
  .def("multiply", &RelayImpl::multiply)
  .def("Function", &RelayImpl::Function)
  .def("Module", &RelayImpl::Module)
  .def("dumpMLIR", &RelayImpl::dumpMLIR)
  .def("print", &RelayImpl::print)
  .def("conv2d", &RelayImpl::conv2d)
  // .def("dumpMLIRLLVM", &RelayImpl::dumpMLIRLLVM)
  // .def("dumpLLVM", &RelayImpl::dumpLLVM)
  .def("Prototype", &RelayImpl::Prototype);
 def("getclass", getclass);
 def("dumpMLIRAffine",dumpMLIRAffine);
 def("inferShape",inferShape);
 def("dumpMLIRLLVM",dumpMLIRLLVM);
 def("dumpLLVM",dumpLLVM);
 class_<mlir::Value>("mlir::Value");
 class_<mlir::FuncOp>("mlir::FuncOp");
 class_<mlir::ModuleOp>("mlir::ModuleOp");
 //class_<mlir::LogicalResult>("mlir::LogicalResult");
}

mlir::Value RelayImpl::var(const char* name, boost::python::list shape,boost::python::list data)
{ 

    PyImport_AppendInittab( "librelay", &initlibrelay); 
  Py_Initialize();
  boost::python::object module = boost::python::import("librelay");
    Location location("fakefile",0,0);
    mlir::Location l= loc(location);
    std::vector<int64_t> dims;
    int num=1;
    for(int i=0;i<len(shape);i++)
    {
        int val = boost::python::call_method<int64_t>(shape.ptr() , "__getitem__" , i);
        num*=val;
        dims.push_back(val);
    }

    std::vector<double> d;
    int length = len(data);
    for(int i=0;i<length;i++)
    {
        int val = boost::python::call_method<double>(data.ptr() , "__getitem__" , i);
        d.push_back(val);
    }
    //std::cout<<"data suc"<<std::endl;

    

    mlir::Type elementType = builder.getF64Type();
    //std::cout<<"builder.getF64Type() 可以执行"<<std::endl;
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    //std::cout<<"mlir::RankedTensorType::get 可以执行"<<std::endl;
    mlir::DenseElementsAttr dataAttribute =  mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(d));
    //std::cout<<"mlir::DenseElementsAttr::get 可以执行"<<std::endl;
    return builder.create<ConstantOp>(l, dataAttribute.getType(), dataAttribute);
}
mlir::Value RelayImpl::transpose(mlir::Value operand)
{ 
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    return builder.create<TransposeOp>(l, operand);
}
void RelayImpl::print(mlir::Value operand)
{ 
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    builder.create<PrintOp>(l, operand);
}
mlir::Value RelayImpl::add(mlir::Value lhs,mlir::Value rhs)
{ 
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    return builder.create<AddOp>(l, lhs, rhs);
}
mlir::Value RelayImpl::multiply(mlir::Value lhs,mlir::Value rhs)
{ 
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    return builder.create<MulOp>(l, lhs, rhs);
}
mlir::Value RelayImpl::conv2d(mlir::Value data,mlir::Value filter/*,int strides_h,int strides_w,int padding*/)
{
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    return builder.create<Conv2dOp>(l, data, filter/*,strides_h,strides_w,padding*/);
}
mlir::FuncOp RelayImpl::Prototype(int number)
{
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    mlir::Type  t =  mlir::UnrankedTensorType::get(builder.getF64Type());
    llvm::SmallVector<mlir::Type, 4> arg_types(number,t);
    llvm::SmallVector<mlir::Type, 4> res_types(1,t);
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    function = mlir::FuncOp::create(l, "main", func_type);
    if (!function)
      return nullptr;
    auto &e = *function.addEntryBlock();
    builder.setInsertionPointToStart(&e);
    entryBlock = &e;
    return function;
}

mlir::FuncOp RelayImpl::Function(boost::python::list free_vars,mlir::Value res)
{ 
    Location location("fakefile",0,0);
    mlir::Location l = loc(location);
    int number = len(free_vars);
    mlir::Type  t =  mlir::UnrankedTensorType::get(builder.getF64Type());
    // llvm::SmallVector<mlir::Type, 4> arg_types(number,t);
    // auto func_type = builder.getFunctionType(arg_types, llvm::None);
    // mlir::FuncOp function = mlir::FuncOp::create(l, "main", func_type);
    // if (!function)
    //   return nullptr;
    // auto &entryBlock = *function.addEntryBlock();
    // builder.setInsertionPointToStart(&entryBlock);
    
    // boost::python::list list;
    // list.append(2);
    // list.append(3);
    // mlir::Value v = var("x",list,"float32");
    builder.create<PrintOp>(l, res);
    ReturnOp returnOp;
    if (!entryBlock->empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock->back());
    if (!returnOp) {
      builder.create<ReturnOp>(l);
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(function.getType().getInputs(),
                                               t));
    }
    //builder.create<ReturnOp>(l,res);
    
    return function;
}

mlir::ModuleOp RelayImpl::Module(mlir::FuncOp func) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    theModule.push_back(func);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the relay operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }
    return theModule;
}

void RelayImpl::dumpMLIR(mlir::ModuleOp module)
{
    
    if (!module) {
        std::cout<<"module is null"<<std::endl;
        return;
    }
    module.dump();
    std::cout<<std::endl;
}
void dumpMLIRAffine(mlir::ModuleOp module)
{
    
    mlir::registerPassManagerCLOptions();
    //mlir::MLIRContext context1;
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::relay::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    pm.addPass(mlir::relay::createLowerToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());

    if (mlir::failed(pm.run(module))){
    std::cout<<"pm.run(module) failed"<<std::endl;
      return ;
    }
    module.dump();
}

void inferShape(mlir::ModuleOp module)
{
    
    mlir::registerPassManagerCLOptions();
    //mlir::MLIRContext context1;
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::relay::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    if (mlir::failed(pm.run(module))){
    std::cout<<"pm.run(module) failed"<<std::endl;
      return ;
    }
    module.dump();
}
void dumpMLIRLLVM(mlir::ModuleOp module)
{
    module.dump();
    mlir::registerPassManagerCLOptions();
    //mlir::MLIRContext context1;
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::relay::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    pm.addPass(mlir::relay::createLowerToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());

    pm.addPass(mlir::relay::createLowerToLLVMPass());

    if (mlir::failed(pm.run(module))){
    std::cout<<"pm.run(module) failed"<<std::endl;
      return ;
    }
    module.dump();
}
void dumpLLVM(mlir::ModuleOp module)
{
  auto llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
       3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return ;
  }
  llvm::errs() << *llvmModule << "\n";
  return ;
}