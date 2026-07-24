#!/bin/bash
# ==============================================================================
# UNIFIED SPECTRE DEPENDENCIES & CONFIGURATION BUILD SCRIPT
# ==============================================================================
# This script installs all communication, math, and data dependencies required
# for SpECTRE on AMD Epyc (Zen) architecture. It then creates an Environment
# Module and a 'configure_spectre.sh' helper script in the SAME directory.
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 0. DYNAMIC DIRECTORY RESOLUTION & VERSION DEFINITIONS
# ------------------------------------------------------------------------------
# Automatically sets the dependency root directory to wherever THIS script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEPS_DIR="${SCRIPT_DIR}"
SRC_DIR="${DEPS_DIR}/src"
MODULES_DIR="${DEPS_DIR}/modulefiles/spectre-deps"

# Explicit Version Definitions
ENV_MODULES_VER="5.4.0"
TCL_VER="8.6.14"
UCX_VER="1.15.0"
OPENMPI_VER="4.1.6"
OPENMPI_MAJOR_MINOR="4.1"
CHARM_VER="v7.0.0"
OPENBLAS_VER="0.3.25"
GSL_VER="2.7.1"
HDF5_VER="1.12.3"
BOOST_VER="1.83.0"
BOOST_VER_UNDERSCORE="1_83_0"
XSIMD_VER="11.1.0"

# Derived Charm++ Installation Root
CHARM_ROOT="${SRC_DIR}/charm-${CHARM_VER}/mpi-linux-x86_64-smp"

# ------------------------------------------------------------------------------
# EXPLANATION OF COMPILER & OPTIMIZATION FLAGS USED THROUGHOUT THIS SCRIPT
# ------------------------------------------------------------------------------
# -O3                       : Enables aggressive compiler optimizations (vectorization, loop unrolling).
# -mavx2                    : Enables 256-bit AVX2 vector instructions for AMD Epyc CPU cores.
# -mfma                     : Enables Fused Multiply-Add (a * b + c in 1 clock cycle) for high-throughput linear algebra.
# -fPIC                     : Generates Position Independent Code, required for dynamic linking & shared objects.
# -Dpthread_yield=sched_yield: Fixes compatibility with legacy libraries calling deprecated pthread_yield().
# -DNDEBUG                  : Suppresses C/C++ runtime assertions for maximum production execution speed.
# ------------------------------------------------------------------------------
OPT_FLAGS="-O3 -mavx2 -mfma -fPIC -Dpthread_yield=sched_yield -DNDEBUG"

# Initial Bootstrap Compilers (System GCC)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export F77=/usr/bin/gfortran

mkdir -p "$SRC_DIR"
mkdir -p "$DEPS_DIR"
mkdir -p "$MODULES_DIR"

cd "$SRC_DIR"

echo "========================================================================="
echo "Starting SpECTRE Dependency Build"
echo "Installation Directory: ${DEPS_DIR}"
echo "========================================================================="

# ==============================================================================
# PHASE 1: COMMUNICATION & RUNTIME STACK
# ==============================================================================

# 1. Tcl (Prerequisite for Environment Modules)
echo ">>> [1/9] Building Tcl ${TCL_VER}..."
if [ ! -f "tcl${TCL_VER}-src.tar.gz" ]; then
    curl -LO "https://prdownloads.sourceforge.net/tcl/tcl${TCL_VER}-src.tar.gz"
fi
if [ ! -d "tcl${TCL_VER}" ]; then
    tar -xzf "tcl${TCL_VER}-src.tar.gz"
    cd "tcl${TCL_VER}/unix"
    ./configure --prefix="${DEPS_DIR}"
    make -j$(nproc)
    make install
    cd ../..
fi
export PATH="${DEPS_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${DEPS_DIR}/lib:${LD_LIBRARY_PATH}"

# 2. Environment Modules
echo ">>> [2/9] Building Environment Modules ${ENV_MODULES_VER}..."
if [ ! -f "modules-${ENV_MODULES_VER}.tar.gz" ]; then
    curl -LO "https://github.com/cea-hpc/modules/releases/download/v${ENV_MODULES_VER}/modules-${ENV_MODULES_VER}.tar.gz"
fi
if [ ! -d "modules-${ENV_MODULES_VER}" ]; then
    tar -xzf "modules-${ENV_MODULES_VER}.tar.gz"
    cd "modules-${ENV_MODULES_VER}"
    ./configure --prefix="${DEPS_DIR}/env-modules" --with-tcl="${DEPS_DIR}/lib"
    make -j$(nproc)
    make install
    cd ..
fi
source "${DEPS_DIR}/env-modules/init/bash"

# Apply Optimization Flags for High-Performance Networking Stack
export CFLAGS="$OPT_FLAGS"
export CXXFLAGS="$OPT_FLAGS"
export FFLAGS="$OPT_FLAGS"
export FCFLAGS="$OPT_FLAGS"

# 3. UCX (Unified Communication X)
# EXPLANATION OF UCX FLAGS:
# --disable-logging / --disable-debug / --disable-assertions: Removes debugging overhead from network drivers.
# --with-verbs --with-mlx5: Direct hardware access to Mellanox InfiniBand verbs and high-speed adapters.
echo ">>> [3/9] Building UCX ${UCX_VER}..."
if [ ! -f "ucx-${UCX_VER}.tar.gz" ]; then
    curl -LO "https://github.com/openucx/ucx/releases/download/v${UCX_VER}/ucx-${UCX_VER}.tar.gz"
fi
if [ ! -d "ucx-${UCX_VER}" ]; then
    tar -xzf "ucx-${UCX_VER}.tar.gz"
    cd "ucx-${UCX_VER}"
    ./configure --prefix="$DEPS_DIR" \
        --disable-logging \
        --disable-debug \
        --disable-assertions \
        --disable-params-check \
        --with-verbs \
        --with-mlx5
    make -j$(nproc)
    make install
    cd ..
fi

# 4. OpenMPI
# EXPLANATION OF OPENMPI FLAGS:
# --with-ucx: Forces OpenMPI to route all point-to-point network communication through UCX.
echo ">>> [4/9] Building OpenMPI ${OPENMPI_VER}..."
if [ ! -f "openmpi-${OPENMPI_VER}.tar.gz" ]; then
    curl -LO "https://download.open-mpi.org/release/open-mpi/v${OPENMPI_MAJOR_MINOR}/openmpi-${OPENMPI_VER}.tar.gz"
fi
if [ ! -d "openmpi-${OPENMPI_VER}" ]; then
    tar -xzf "openmpi-${OPENMPI_VER}.tar.gz"
    cd "openmpi-${OPENMPI_VER}"
    ./configure --prefix="$DEPS_DIR" --with-ucx="$DEPS_DIR"
    make -j$(nproc)
    make install
    cd ..
fi

# Route all future compiles through our custom OpenMPI wrappers
export PATH="${DEPS_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${DEPS_DIR}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${DEPS_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CC="${DEPS_DIR}/bin/mpicc"
export CXX="${DEPS_DIR}/bin/mpicxx"
export FC="${DEPS_DIR}/bin/mpifort"
export F77="${DEPS_DIR}/bin/mpifort"

# 5. Charm++
# EXPLANATION OF CHARM++ BUILD COMMAND:
# mpi-linux-x86_64: Sets underlying network layer to MPI for cross-node messaging.
# smp              : Enables multi-threading support (Shared Memory Multiprocessing) within each node.
# --with-production: Builds with -O3 optimization and strips out verbose Charm++ internal tracing/debug code.
echo ">>> [5/9] Building Charm++ ${CHARM_VER}..."

# Only build if the final compiled wrapper doesn't exist yet
if [ ! -f "${CHARM_ROOT}/bin/charmc" ]; then
    if [ ! -d "charm-${CHARM_VER}" ]; then
        git clone https://github.com/UIUC-PPL/charm.git "charm-${CHARM_VER}"
    fi
    cd "charm-${CHARM_VER}"
    git fetch
    git checkout ${CHARM_VER}
    
    ./build charm++ mpi-linux-x86_64 smp --with-production -j$(nproc)
    
    # ==============================================================================
    # CHARM++ POST-BUILD PATCHES
    # ==============================================================================
    echo ">>> Applying fixes to compiled Charm++..."
    
    # 1. Fix invalid C++ template constructor syntax rejected by GCC 11+
    sed -i 's/CkHashtableAdaptorT<T>(/CkHashtableAdaptorT(/g' "${CHARM_ROOT}/include/ckhashtable.h"
    
    # 2. Strip the initial-exec TLS model so Python can dynamically load shared libraries
    find "${CHARM_ROOT}" -type f -exec sed -i 's/-ftls-model=initial-exec//g' {} +
    
    cd ..
else
    echo "Charm++ ${CHARM_VER} is already built. Skipping."
fi

# ==============================================================================
# PHASE 2: MATH & DATA LIBRARIES
# ==============================================================================

# 6. OpenBLAS
# EXPLANATION OF OPENBLAS FLAGS:
# TARGET=ZEN    : Compiles hand-optimized assembly kernels for AMD Zen microarchitecture.
# USE_OPENMP=0  : Disables internal OpenBLAS threading to prevent CPU oversubscription with Charm++ threads.
# DYNAMIC_ARCH=0: Focuses compilation exclusively on AMD Zen instructions rather than multi-target dispatching.
echo ">>> [6/9] Building OpenBLAS ${OPENBLAS_VER}..."
if [ ! -f "OpenBLAS-${OPENBLAS_VER}.tar.gz" ]; then
    curl -LO "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VER}/OpenBLAS-${OPENBLAS_VER}.tar.gz"
fi
if [ ! -d "OpenBLAS-${OPENBLAS_VER}" ]; then
    tar -xzf "OpenBLAS-${OPENBLAS_VER}.tar.gz"
    cd "OpenBLAS-${OPENBLAS_VER}"
    make CC=$CC FC=$FC CFLAGS="$CFLAGS" FFLAGS="$FFLAGS" USE_OPENMP=0 DYNAMIC_ARCH=0 TARGET=ZEN -j$(nproc)
    make PREFIX="$DEPS_DIR" install
    cd ..
fi

# 7. GSL (GNU Scientific Library)
echo ">>> [7/9] Building GSL ${GSL_VER}..."
if [ ! -f "gsl-${GSL_VER}.tar.gz" ]; then
    curl -LO "https://ftp.gnu.org/gnu/gsl/gsl-${GSL_VER}.tar.gz"
fi
if [ ! -d "gsl-${GSL_VER}" ]; then
    tar -xzf "gsl-${GSL_VER}.tar.gz"
    cd "gsl-${GSL_VER}"
    ./configure --prefix="$DEPS_DIR" CC=$CC
    make -j$(nproc)
    make install
    cd ..
fi

# 8. HDF5 & Boost
# EXPLANATION OF HDF5 / BOOST FLAGS:
# --enable-parallel : Compiles HDF5 with Parallel MPI-IO support using our custom mpicc wrapper.
# link=shared       : Builds Boost as shared objects (.so) required by SpECTRE runtime loaders.
echo ">>> [8/9] Building HDF5 ${HDF5_VER} and Boost ${BOOST_VER}..."
if [ ! -f "hdf5-${HDF5_VER}.tar.gz" ]; then
    HDF5_MAJOR_MINOR=$(echo ${HDF5_VER} | cut -d. -f1,2)
    curl -LO "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR_MINOR}/hdf5-${HDF5_VER}/src/hdf5-${HDF5_VER}.tar.gz"
fi
if [ ! -d "hdf5-${HDF5_VER}" ]; then
    tar -xzf "hdf5-${HDF5_VER}.tar.gz"
    cd "hdf5-${HDF5_VER}"
    ./configure --prefix="$DEPS_DIR" --enable-parallel --enable-shared CC=$CC CXX=$CXX
    make -j$(nproc)
    make install
    cd ..
fi

if [ ! -f "boost_${BOOST_VER_UNDERSCORE}.tar.gz" ]; then
    curl -L -o "boost_${BOOST_VER_UNDERSCORE}.tar.gz" "https://sourceforge.net/projects/boost/files/boost/${BOOST_VER}/boost_${BOOST_VER_UNDERSCORE}.tar.gz/download"
fi
if [ ! -d "boost_${BOOST_VER_UNDERSCORE}" ]; then
    tar -xzf "boost_${BOOST_VER_UNDERSCORE}.tar.gz"
    cd "boost_${BOOST_VER_UNDERSCORE}"
    ./bootstrap.sh --prefix="$DEPS_DIR"
    ./b2 install -j$(nproc) variant=release link=shared threading=multi cxxflags="$CXXFLAGS"
    cd ..
fi

# ==============================================================================
# 9. XSIMD (C++ wrapper for SIMD intrinsics - Required for -mavx2)
# ==============================================================================
echo ">>> [9/9] Building XSIMD ${XSIMD_VER}..."
if [ ! -f "xsimd-${XSIMD_VER}.tar.gz" ]; then
    # FIXED: Removed the 'v' before ${XSIMD_VER} in the URL
    curl -L -o "xsimd-${XSIMD_VER}.tar.gz" "https://github.com/xtensor-stack/xsimd/archive/refs/tags/${XSIMD_VER}.tar.gz"
fi
if [ ! -d "xsimd-${XSIMD_VER}" ]; then
    tar -xzf "xsimd-${XSIMD_VER}.tar.gz"
    cd "xsimd-${XSIMD_VER}"
    # Header-only library; CMake just configures the install paths
    cmake -D CMAKE_INSTALL_PREFIX="$DEPS_DIR" .
    make install
    cd ..
fi

# ==============================================================================
# PHASE 3: ENVIRONMENT MODULEFILE GENERATION
# ==============================================================================
echo "========================================================================="
echo "Generating Environment Modulefile"
echo "========================================================================="

cat << EOF > "${MODULES_DIR}/default"
#%Module1.0
proc ModulesHelp { } {
    puts stderr "Loads SpECTRE custom dependencies with AMD Epyc optimized flags"
}

module-whatis "SpECTRE optimized dependencies"

set prefix ${DEPS_DIR}

prepend-path PATH \$prefix/bin
prepend-path LD_LIBRARY_PATH \$prefix/lib
prepend-path CPATH \$prefix/include
prepend-path PKG_CONFIG_PATH \$prefix/lib/pkgconfig

setenv CHARM_ROOT ${CHARM_ROOT}
EOF

# ==============================================================================
# PHASE 4: SPECTRE CONFIGURE HELPER SCRIPT GENERATION
# ==============================================================================
# Generates 'configure_spectre.sh' directly in ${DEPS_DIR}
CONFIG_HELPER="${DEPS_DIR}/configure_spectre.sh"

echo "========================================================================="
echo "Generating Configuration Helper: ${CONFIG_HELPER}"
echo "========================================================================="

cat << EOF > "${CONFIG_HELPER}"
#!/bin/bash
# ==============================================================================
# SPECTRE CMAKE CONFIGURATION HELPER SCRIPT
# ==============================================================================
# Run this script inside your SpECTRE build directory (e.g., spectre/build).
# ==============================================================================

set -e

# Load custom environment module generated during dependency build
module purge > /dev/null 2>&1 || true
source "${DEPS_DIR}/env-modules/init/bash"
module use "${DEPS_DIR}/modulefiles"
module load spectre-deps/default

OPT_FLAGS="-O3 -mavx2 -mfma -fPIC -Dpthread_yield=sched_yield -DNDEBUG"

# EXPLANATION OF CMAKE CONFIGURATION FLAGS:
# -D CMAKE_BUILD_TYPE=Release          : Enables full compiler optimization pipeline.
# -D CHARM_ROOT                        : Absolute path to our custom SMP Charm++ build.
# -D CMAKE_PREFIX_PATH                 : Directs CMake to find OpenBLAS, GSL, Boost, and HDF5 in our custom dir.
# -D MEMORY_ALLOCATOR=SYSTEM           : Uses system glibc malloc/free (prevents allocator collisions with Charm++).
# -D USE_XSIMD=ON                      : Enables SpECTRE's high-level SIMD vectorization abstractions.
# -D USE_LTO=OFF                       : Disables Link-Time Optimization during build to prevent huge link times/RAM exhaustion.
# -D DEBUG_SYMBOLS=ON                  : Retains debug symbol tables (-g) for stack backtraces (e.g. GDB profiling).
# -D SPECTRE_FETCH_MISSING_DEPS=ON     : Automatically downloads minor header-only C++ dependencies (Blaze, Brigand, Catch2).

cmake -D CMAKE_C_COMPILER="gcc" \\
      -D CMAKE_CXX_COMPILER="g++" \\
      -D CMAKE_Fortran_COMPILER="gfortran" \\
      -D CHARM_ROOT="${CHARM_ROOT}" \\
      -D CMAKE_C_FLAGS_RELEASE="\${OPT_FLAGS}" \\
      -D CMAKE_CXX_FLAGS_RELEASE="\${OPT_FLAGS}" \\
      -D CMAKE_Fortran_FLAGS_RELEASE="\${OPT_FLAGS}" \\
      -D CMAKE_BUILD_TYPE=Release \\
      -D CMAKE_PREFIX_PATH="${DEPS_DIR}" \\
      -D BOOST_ROOT="${DEPS_DIR}" \\
      -D HDF5_ROOT="${DEPS_DIR}" \\
      -D GSL_ROOT_DIR="${DEPS_DIR}" \\
      -D BLA_VENDOR=OpenBLAS \\
      -D BLAS_DIR="${DEPS_DIR}" \\
      -D LAPACK_DIR="${DEPS_DIR}" \\
      -D Python_EXECUTABLE=\$(which python3) \\
      -D MEMORY_ALLOCATOR=SYSTEM \\
      -D BUILD_PYTHON_BINDINGS=ON \\
      -D ENABLE_PARAVIEW=OFF \\
      -D USE_XSIMD=ON \\
      -D USE_LTO=OFF \\
      -D DEBUG_SYMBOLS=ON \\
      -D MACHINE=Sonic \\
      -D SPECTRE_FETCH_MISSING_DEPS=ON \\
      "\$@"
EOF

chmod +x "${CONFIG_HELPER}"

# ==============================================================================
# INSTRUCTIONS FOR USE
# ==============================================================================
cat << EOF

=========================================================================
BUILD COMPLETE! All dependencies and configuration files are ready.
=========================================================================

Location of Dependencies : ${DEPS_DIR}
Location of Configure    : ${CONFIG_HELPER}

-------------------------------------------------------------------------
HOW TO USE THIS SETUP TO BUILD SPECTRE:
-------------------------------------------------------------------------

Step 1: Create and enter your desired build directory (anywhere on the system):
        mkdir -p /path/to/your/build_dir && cd /path/to/your/build_dir

Step 2: Execute the configuration helper script, passing the path to 
        your SpECTRE source directory:
        ${CONFIG_HELPER} /path/to/spectre_source

Step 3: Build your target executable (e.g., Valencia GRMHD):
        make -j\$(nproc) EvolveValenciaDivCleanWithHorizon

-------------------------------------------------------------------------
RE-LOADING THE ENVIRONMENT IN SLURM BENCHMARK SCRIPTS:
-------------------------------------------------------------------------
Add these lines to your Slurm job submission scripts:

source ${DEPS_DIR}/env-modules/init/bash
module use ${DEPS_DIR}/modulefiles
module load spectre-deps/default
=========================================================================
EOF