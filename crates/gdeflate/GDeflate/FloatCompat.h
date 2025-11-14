#pragma once

// NVCC's device compilation still lacks native support for the C2x
// extended floating-point keywords that glibc assumes are available
// whenever `__GNUC_PREREQ(13, 0)` succeeds.  Instead of attempting to
// redefine those types here (which interferes with glibc's own template
// specialisations), we shim `bits/floatn-common.h` to momentarily clamp
// `__GNUC_PREREQ` while the glibc fallbacks are being parsed.  Keeping
// this header available ensures the build-script's `-include` flag remains
// valid even though the compatibility logic now lives in that wrapper.

#if !defined(__RG_FLOATCOMPAT_HOOKS_ENABLED)
#  if defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDABE__)
#    define __RG_FLOATCOMPAT_HOOKS_ENABLED 1
#  elif defined(__has_include)
#    if __has_include(<cuda_runtime_api.h>)
#      define __RG_FLOATCOMPAT_HOOKS_ENABLED 1
#    endif
#  endif
#endif

#if defined(__RG_FLOATCOMPAT_HOOKS_ENABLED)

#ifndef __RG_NVCC_DISABLE_FAKE_BUILTINS
#define __RG_NVCC_DISABLE_FAKE_BUILTINS 1

#if !defined(__CUDA_ARCH__)
#  include <cuda_runtime_api.h>
#  include "crt/host_runtime.h"
#  include "FloatCompatLaunchPatch.h"

#endif  // !__CUDA_ARCH__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)

#  ifndef __noinline__
#    if defined(_MSC_VER)
#      define __noinline__ __declspec(noinline)
#    elif defined(__GNUC__)
#      define __noinline__ __attribute__((noinline))
#    else
#      define __noinline__
#    endif
#  endif

#endif  // !__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#ifdef __cplusplus

#  ifdef __NVCC__
#    pragma message("FloatCompat: __NVCC__ defined")
#  else
#    pragma message("FloatCompat: __NVCC__ NOT defined")
#  endif
#  ifdef __CUDACC__
#    pragma message("FloatCompat: __CUDACC__ defined")
#  endif
#  ifdef __CUDA_ARCH__
#    pragma message("FloatCompat: __CUDA_ARCH__ defined")
#  else
#    pragma message("FloatCompat: __CUDA_ARCH__ NOT defined")
#  endif
#  ifdef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#    pragma message("FloatCompat: internal compiler headers")
#  endif
#  ifdef __CUDA_EXEC_CHECK_DISABLE__
#    pragma message("FloatCompat: exec check disable")
#  endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

/*
 * The compatibility shims below are only needed when the translation unit is
 * being processed by the host compiler that NVCC dispatches to.  During the
 * stub generation / device front-end passes, NVCC defines
 * `__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__` while its internal headers are
 * on the include stack, and we must avoid replacing the builtins in that
 * situation because the internal headers rely on the real definitions.
 */

#  if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)

namespace rg_nvcc_compat {

template<bool B>
struct bool_constant {
	static constexpr bool value = B;
};

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<typename...>
struct disjunction : false_type {};

template<typename B1>
struct disjunction<B1> : bool_constant<B1::value> {};

template<typename B1, typename... Bn>
struct disjunction<B1, Bn...>
	: bool_constant<B1::value || disjunction<Bn...>::value> {};

template<typename...>
using void_t = void;

template<typename T>
struct remove_cv {
	using type = T;
};

template<typename T>
struct remove_cv<const T> {
	using type = T;
};

template<typename T>
struct remove_cv<volatile T> {
	using type = T;
};

template<typename T>
struct remove_cv<const volatile T> {
	using type = T;
};

template<typename T>
using remove_cv_t = typename remove_cv<T>::type;

template<typename T>
struct remove_reference {
	using type = T;
};

template<typename T>
struct remove_reference<T&> {
	using type = T;
};

template<typename T>
struct remove_reference<T&&> {
	using type = T;
};

template<typename T>
using remove_reference_t = typename remove_reference<T>::type;

template<typename T>
struct add_rvalue_reference {
	using type = T&&;
};

template<>
struct add_rvalue_reference<void> {
	using type = void;
};

template<>
struct add_rvalue_reference<const void> {
	using type = const void;
};

template<>
struct add_rvalue_reference<volatile void> {
	using type = volatile void;
};

template<>
struct add_rvalue_reference<const volatile void> {
	using type = const volatile void;
};

template<typename T>
typename add_rvalue_reference<T>::type declval() noexcept;

using size_t = decltype(sizeof(0));

template<typename T>
struct is_void : false_type {};

template<>
struct is_void<void> : true_type {};

template<>
struct is_void<const void> : true_type {};

template<>
struct is_void<volatile void> : true_type {};

template<>
struct is_void<const volatile void> : true_type {};

template<typename T>
struct is_array : false_type {};

template<typename T, size_t N>
struct is_array<T[N]> : true_type {};

template<typename T>
struct is_array<T[]> : true_type {};

#if defined(__has_builtin)
#  if __has_builtin(__is_function)
template<typename T>
struct is_function : bool_constant<__is_function(T)> {};
#  else
template<typename T>
struct is_function : false_type {};
#  endif
#else
template<typename T>
struct is_function : false_type {};
#endif

template<typename From, typename To, bool = disjunction<is_void<From>, is_function<To>, is_array<To>>::value>
struct is_convertible_helper {
	using type = bool_constant<is_void<To>::value>;
};

template<typename From, typename To>
class is_convertible_helper<From, To, false> {
	private:
		static void test_aux(To) noexcept;

		template<typename From1, typename To1,
		         typename = decltype(test_aux(declval<From1>()))>
		static true_type test(int);

		template<typename, typename>
		static false_type test(...);

	public:
		using type = decltype(test<From, To>(0));
};

template<typename From, typename To>
struct is_convertible : is_convertible_helper<From, To>::type {};

template<typename From, typename To, bool = disjunction<is_void<From>, is_function<To>, is_array<To>>::value>
struct is_nothrow_convertible_helper {
	using type = is_void<To>;
};

template<typename From, typename To>
class is_nothrow_convertible_helper<From, To, false> {
	private:
		static void test_aux(To) noexcept;

		template<typename From1, typename To1>
		static bool_constant<noexcept(test_aux(declval<From1>()))>
		test(int);

		template<typename, typename>
		static false_type test(...);

	public:
		using type = decltype(test<From, To>(0));
};

template<typename From, typename To>
struct is_nothrow_convertible : is_nothrow_convertible_helper<From, To>::type {};

}  // namespace rg_nvcc_compat

#  ifndef __is_convertible
#    define __is_convertible(_From, _To) \
			(::rg_nvcc_compat::is_convertible<_From, _To>::value)
#  endif
#  ifndef __is_nothrow_convertible
#    define __is_nothrow_convertible(_From, _To) \
			(::rg_nvcc_compat::is_nothrow_convertible<_From, _To>::value)
#  endif
#  ifndef __remove_cv
#    define __remove_cv(...) ::rg_nvcc_compat::remove_cv_t<__VA_ARGS__>
#  endif
#  ifndef __remove_reference
#    define __remove_reference(...) \
			::rg_nvcc_compat::remove_reference_t<__VA_ARGS__>
#  endif
#  ifndef __reference_constructs_from_temporary
#    define __reference_constructs_from_temporary(...) 0
#  endif
#  ifndef __reference_converts_from_temporary
#    define __reference_converts_from_temporary(...) 0
#  endif
#  ifndef __is_layout_compatible
#    define __is_layout_compatible(...) 0
#  endif
#  ifndef __is_pointer_interconvertible_base_of
#    define __is_pointer_interconvertible_base_of(...) 0
#  endif

#  endif  // !__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#else
#  pragma message("FloatCompat: included in non-C++ TU")
#endif  // __cplusplus

#endif  // __RG_NVCC_DISABLE_FAKE_BUILTINS

#endif  // defined(__RG_FLOATCOMPAT_HOOKS_ENABLED)

#if defined(__RG_FLOATCOMPAT_HOOKS_ENABLED)

#ifndef CUDA_FLOAT_COMPAT_DEFINED
#define CUDA_FLOAT_COMPAT_DEFINED

// Intentionally empty: see the note above.  The real adjustments happen in
// `GDeflate/features.h` which is picked up before the system `<features.h>`
// thanks to the include path ordering established in `build.rs`.

#endif  // CUDA_FLOAT_COMPAT_DEFINED

#endif  // defined(__RG_FLOATCOMPAT_HOOKS_ENABLED)
