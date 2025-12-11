export PATH=""
export CFLAGS=""
export CXXFLAGS=""
export LD_LIBRARY_PATH=""

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_HOME="/lustre_archive/apps/correlator/cuda-11.7"
export CUDA_PATH="/lustre_archive/apps/correlator/cuda-11.7"
export CUDA_INSTALL_PATH="/lustre_archive/apps/correlator/cuda-11.7"

export PSRSOFT="/lustre_archive/apps/tdsoft/usr"
export PSRCAT_FILE="/lustre_archive/apps/tdsoft/usr/share/psrcat/psrcat.db"

export TEMPO="$PSRSOFT/src/tempo"
export PRESTO="$PSRSOFT/src/presto_old"
export TEMPO2="$PSRSOFT/src/tempo2"
export SIGPROC="$PSRSOFT/src/sixproc"

export PGPLOT_DEV="/xwindow"
export PGPLOT_DIR="$PSRSOFT/src/pgplot"
export PGPLOT_FONT="$PGPLOT_DIR/grfont.dat"

export LD_LIBRARY_PATH="/lustre_archive/apps/correlator/cuda-11.7/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre_archive/apps/tdsoft/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre_archive/apps/tdsoft/usr/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre_archive/apps/correlator/mpi/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre_archive/apps/astrosoft/Offline/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PRESTO/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PSRSOFT/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PSRSOFT/src/pgplot"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre_archive/apps/tdsoft/jre1.8.0_441/lib/amd64/server"

export PATH="/lustre_archive/apps/tdsoft/bin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/usr/bin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/conda/bin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/conda/condabin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/jre1.8.0_441/bin"
export PATH="$PATH:/lustre_archive/apps/correlator/cuda-11.7/bin"
export PATH="$PATH:/lustre_archive/apps/correlator/mpi/bin"
export PATH="$PATH:/lustre_archive/apps/astrosoft/Offline/bin"
export PATH="$PATH:/lustre_archive/gnsmdev/.local/bin"
export PATH="$PATH:/usr/local/bin"
export PATH="$PATH:/usr/bin"
export PATH="$PATH:/usr/local/sbin"
export PATH="$PATH:/usr/sbin"
export PATH="$HOME/.local/bin:$PATH"

export PATH="$PATH:$PSRSOFT/bin"
export PATH="$PATH:$PRESTO/bin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/plotres"
export PATH="$PATH:/lustre_archive/apps/tdsoft/usr/src/fake_simulation/bin"
export PATH="$PATH:/lustre_archive/apps/tdsoft/usr/src/ffancy"
export PATH="$PATH:/usr/local/cuda/bin"

export PIPX_HOME="/lustre_archive/apps/tdsoft/pipx"
export PIPX_BIN_DIR="/lustre_archive/apps/tdsoft/bin"
export PIPX_MAN_DIR="/lustre_archive/apps/tdsoft/man"
export PIPX_GLOBAL_HOME="/lustre_archive/apps/tdsoft/pipx"
export PIPX_GLOBAL_BIN_DIR="/lustre_archive/apps/tdsoft/bin"

export TDSOFT="/lustre_archive/apps/tdsoft"
export SPOTLIGHT_DATA="/lustre_archive/spotlight/data"

export VER0DIR="${TDSOFT}/ver0"
export PSS_VER0_DIR="/lustre_archive/spotlight/data/Pulsar_Search_Script"
export PULSELINE_VER0_DIR="/lustre_data/spotlight/data/pulsar_search_pipeline_ver0"
export PULSELINE_DEV_DIR="/lustre_data/spotlight/data/pulsar_search_pipeline_dev"
export PULSELINE_VER0_TEST_DIR="/lustre_archive/spotlight/data/pulsar_search_pipeline_ver0"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CUDA_COMPUTE_CAPABILITIES="8.0"
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=1

alias rfi_filter="python /lustre_archive/spotlight/raghav/PulsarX/scripts/multi_rfi_filter_filtool.py"

# <<< conda initialize <<<
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre_archive/apps/tdsoft/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre_archive/apps/tdsoft/conda/etc/profile.d/conda.sh" ]; then
        . "/lustre_archive/apps/tdsoft/conda/etc/profile.d/conda.sh"
    else
        export PATH="/lustre_archive/apps/tdsoft/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
