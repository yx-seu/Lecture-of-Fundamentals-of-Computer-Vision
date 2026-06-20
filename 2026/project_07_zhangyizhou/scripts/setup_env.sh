#!/bin/bash
# Environment setup for Vivado/Vitis 2025.2
export VITIS_DIR=/home/aika/AMD/2025.2/Vitis
export VIVADO_DIR=/home/aika/AMD/2025.2/Vivado
export LD_LIBRARY_PATH=$VITIS_DIR/lib/lnx64.o:$VITIS_DIR/bin/unwrapped/lnx64.o:$LD_LIBRARY_PATH
export PATH=$VIVADO_DIR/bin:$VITIS_DIR/bin:$PATH
echo "Vivado/Vitis 2025.2 environment ready"
