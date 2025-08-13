# Internal Utilities

Internal utility functions and algorithms used throughout FiniteDiff.jl. These functions are primarily for advanced users who need to understand or extend the library's functionality.

## Array Utilities

Internal utilities for handling different array types and ensuring compatibility across the Julia ecosystem:

### Type Conversion Functions

These functions help FiniteDiff.jl work with various array types including `StaticArrays`, structured matrices, and GPU arrays:

- **`_vec(x)`**: Vectorizes arrays while preserving scalars unchanged
- **`_mat(x)`**: Ensures matrix format, converting vectors to column matrices  
- **`setindex(x, v, i...)`**: Non-mutating setindex operations for immutable arrays

### Hessian Utilities

Helper functions specifically for Hessian computation:

- **`_hessian_inplace(x)`**: Determines whether to use in-place operations based on array mutability
- **`__Symmetric(x)`**: Wraps matrices in `Symmetric` views for mathematical correctness
- **`mutable_zeromatrix(x)`**: Creates mutable zero matrices compatible with input arrays

## Sparse Jacobian Internals

Graph coloring and sparse matrix algorithms for efficient Jacobian computation:

### Matrix Construction

- **`_make_Ji(...)`**: Constructs Jacobian contribution matrices for both sparse and dense cases
- **`_colorediteration!(...)`**: Core loop for sparse Jacobian assembly using graph coloring
- **`_findstructralnz(A)`**: Finds structural non-zero patterns in dense matrices

### Sparsity Detection

- **`_use_findstructralnz(sparsity)`**: Determines when to use structural sparsity information
- **`_use_sparseCSC_common_sparsity(J, sparsity)`**: Tests for common sparsity patterns between matrices

### Performance Optimizations

- **`fast_jacobian_setindex!(...)`**: Optimized index operations for sparse Jacobian assembly
- **`void_setindex!(...)`**: Wrapper for setindex operations that discards return values

## JVP Utilities

- **`resize!(cache::JVPCache, i)`**: Resizes JVP cache arrays for dynamic problems
- **`resize!(cache::JacobianCache, i)`**: Resizes Jacobian cache arrays for dynamic problems

## Error Handling

- **`fdtype_error(::Type{T})`**: Provides informative error messages for unsupported finite difference type combinations

## Design Philosophy

These internal utilities follow several design principles:

1. **Type Stability**: All functions are designed for type-stable operations
2. **Zero Allocation**: Internal utilities avoid allocations when possible
3. **Genericity**: Support for multiple array types (dense, sparse, static, GPU)
4. **Performance**: Optimized for the specific needs of finite difference computation
5. **Safety**: Proper error handling and bounds checking where needed

## Advanced Usage

These functions are primarily internal, but advanced users may find them useful for:

- **Custom finite difference implementations**
- **Integration with other differentiation libraries**  
- **Performance optimization in specialized applications**
- **Understanding the implementation details of FiniteDiff.jl**

Most users should rely on the main API functions rather than calling these utilities directly.