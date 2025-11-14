#define __has_builtin(x) 0
#if __has_builtin(__is_convertible)
#error builtin exists
#endif
