#define __is_convertible(...) 0
#if __has_builtin(__is_convertible)
#error builtin exists
#endif
