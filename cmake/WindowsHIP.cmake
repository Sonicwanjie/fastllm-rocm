# Windows HIP compilation helper
# Compiles .hip files to .obj files using hipcc
# The .obj files are then passed to the MSVC linker

set(HIPCC_COMPILER "C:/rocm/bin/hipcc.exe" CACHE STRING "HIP compiler path")
set(ROCM_INCLUDE_DIR "C:/rocm/include" CACHE STRING "ROCm include directory")
set(HIP_ARCH "gfx1151" CACHE STRING "HIP target architecture")

function(hip_compile_objects OUTPUT_VAR)
    set(OBJ_FILES "")
    foreach(HIP_FILE ${ARGN})
        get_filename_component(HIP_ABS "${HIP_FILE}" ABSOLUTE)
        get_filename_component(HIP_DIR "${HIP_ABS}" DIRECTORY)
        get_filename_component(HIP_NAME "${HIP_FILE}" NAME_WE)
        
        # Create unique obj name including relative path to avoid collisions
        file(RELATIVE_PATH HIP_REL "${CMAKE_SOURCE_DIR}" "${HIP_ABS}")
        string(REPLACE "/" "_" HIP_OBJ_NAME "${HIP_REL}")
        string(REPLACE "." "_" HIP_OBJ_NAME "${HIP_OBJ_NAME}")
        
        set(OBJ_FILE "${CMAKE_BINARY_DIR}/hip_obj/${HIP_OBJ_NAME}.obj")
        
        # Build include paths from source tree
        set(HIP_INCLUDES
            -I "${CMAKE_SOURCE_DIR}/include"
            -I "${CMAKE_SOURCE_DIR}/include/devices/cuda"
            -I "${CMAKE_SOURCE_DIR}/include/utils"
            -I "${CMAKE_SOURCE_DIR}/include/models"
            -I "${CMAKE_SOURCE_DIR}/include/blocks"
            -I "${CMAKE_SOURCE_DIR}/include/devices/cpu"
            -I "${CMAKE_SOURCE_DIR}/include/devices/disk"
            -I "${CMAKE_SOURCE_DIR}/third_party/json11"
            -I "${CMAKE_SOURCE_DIR}/third_party/gguf"
            -I "${CMAKE_SOURCE_DIR}/third_party/flashinfer"
            -I "${ROCM_INCLUDE_DIR}"
            -I "${ROCM_INCLUDE_DIR}/hip"
            -I "${ROCM_INCLUDE_DIR}/hipblas"
            -I "${ROCM_INCLUDE_DIR}/rocprim"
            -I "${ROCM_INCLUDE_DIR}/rocwmma"
            -I "${HIP_DIR}"
        )
        
        set(HIP_DEFINES
            -DUSE_ROCM -DUSE_CUDA -DHIPBLAS_V2
        )
        
        add_custom_command(
            OUTPUT "${OBJ_FILE}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/hip_obj"
            COMMAND "${HIPCC_COMPILER}"
                ${HIP_DEFINES}
                ${HIP_INCLUDES}
                --offload-arch=${HIP_ARCH}
                -O3 -std=c++17
                -c "${HIP_ABS}"
                -o "${OBJ_FILE}"
            MAIN_DEPENDENCY "${HIP_ABS}"
            COMMENT "[HIP] ${HIP_REL}"
        )
        list(APPEND OBJ_FILES "${OBJ_FILE}")
    endforeach()
    set(${OUTPUT_VAR} ${OBJ_FILES} PARENT_SCOPE)
endfunction()