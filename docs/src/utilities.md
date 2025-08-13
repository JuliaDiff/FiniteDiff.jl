# Utilities

Internal utility functions and types that may be useful for advanced users.

## Step Size Computation

While these functions are primarily internal, they can be useful for understanding and customizing finite difference computations:

### Default Step Sizes

Different finite difference methods use different optimal step sizes to balance truncation error and round-off error:

- **Forward differences**: `sqrt(eps(T))` - balances O(h) truncation error with round-off
- **Central differences**: `cbrt(eps(T))` - balances O(h²) truncation error with round-off  
- **Hessian central**: `eps(T)^(1/4)` - balances O(h²) truncation error for second derivatives

### Complex Step Differentiation

Complex step differentiation uses machine epsilon since it avoids subtractive cancellation:

⚠️ **Important**: `f` must be a function of a real variable that is also complex analytic when the input is complex!

## Array Utilities

Internal utilities for handling different array types and ensuring compatibility:

- `_vec(x)`: Vectorizes arrays while preserving scalars
- `_mat(x)`: Ensures matrix format, converting vectors to column matrices
- `setindex(x, v, i...)`: Non-mutating setindex for immutable arrays

These functions help FiniteDiff.jl work with various array types including `StaticArrays`, structured matrices, and GPU arrays.