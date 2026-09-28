// Deliberately just this include: Tensor.h is header-only (every member is
// inline), so this translation unit exists solely to catch header errors
// that only surface when compiled standalone (missing includes, unguarded
// macros), it produces no additional symbols.
#include <deepity/backend/Tensor.h>
