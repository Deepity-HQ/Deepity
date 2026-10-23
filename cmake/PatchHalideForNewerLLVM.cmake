# Halide v21.0.0's NVPTX/CUDA codegen (which Deepity never uses -- only
# host/X86 AOT generation) references llvm::TargetOptions::UnsafeFPMath,
# removed from LLVM without replacement sometime after LLVM 21. Halide's
# own upstream main branch already deletes these same two assignments
# outright (confirmed against github.com/halide/Halide@main); this just
# applies that same fix to the pinned v21.0.0 tag so it builds against a
# newer system LLVM (this repo built it against 22.1.8). Pure CMake (no
# sed/patch binary) so this runs identically on every platform FetchContent
# supports, including Windows.
#
# DEEPITY_HALIDE_SRC_DIR is passed in via -D by the PATCH_COMMAND invocation
# in CMakeLists.txt -- deliberately not relying on whatever working
# directory PATCH_COMMAND happens to run in, which FetchContent doesn't
# document as stable.
if(NOT DEEPITY_HALIDE_SRC_DIR)
    message(FATAL_ERROR "DEEPITY_HALIDE_SRC_DIR must be passed via -D")
endif()

foreach(_deepity_halide_patch_file CodeGen_PTX_Dev.cpp CodeGen_Internal.cpp)
    set(_deepity_halide_patch_path "${DEEPITY_HALIDE_SRC_DIR}/src/${_deepity_halide_patch_file}")
    file(READ "${_deepity_halide_patch_path}" _deepity_halide_patch_content)
    string(REPLACE "    options.UnsafeFPMath = true;\n" ""
        _deepity_halide_patch_content "${_deepity_halide_patch_content}")
    string(REPLACE "    options.UnsafeFPMath = !per_instruction_fast_math_flags;\n" ""
        _deepity_halide_patch_content "${_deepity_halide_patch_content}")
    file(WRITE "${_deepity_halide_patch_path}" "${_deepity_halide_patch_content}")
endforeach()
