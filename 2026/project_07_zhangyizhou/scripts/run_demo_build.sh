#!/bin/bash
#=============================================================================
# run_demo_build.sh — Complete LeNet-5 Demo Build
#=============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " LeNet-5 MNIST Demo - Complete Build"
echo " Alinx AX7010 (xc7z010clg400-1)"
echo "============================================"
echo ""

#---------------------------------------------------------------------
# Step 0: Verify prerequisites
#---------------------------------------------------------------------
echo "[0/3] Checking prerequisites..."

VIVADO_DIR=/home/aika/AMD/2025.2/Vivado
VITIS_DIR=/home/aika/AMD/2025.2/Vitis

if [ ! -d "$VIVADO_DIR" ]; then
    echo "ERROR: Vivado not found at $VIVADO_DIR"
    exit 1
fi

# Source Vivado
source $VIVADO_DIR/settings64.sh
source $VITIS_DIR/settings64.sh

# Check HLS IP exists
HLS_IP=$PROJ_DIR/hls/lenet5_accel/lenet5_accel/solution4/impl/ip/component.xml
if [ ! -f "$HLS_IP" ]; then
    echo "ERROR: HLS IP not found. Run HLS first:"
    echo "  cd $PROJ_DIR/hls/lenet5_accel && vitis-run --tcl run.tcl --part xc7z010clg400-1"
    exit 1
fi
echo "  HLS IP: OK"
echo "  Vivado: $(vivado -version | head -1)"
echo ""

#---------------------------------------------------------------------
# Step 1: Vivado Synthesis + Implementation + Bitstream + XSA
#---------------------------------------------------------------------
echo "[1/3] Running Vivado build..."
echo "  This will take 30-60 minutes. Progress is logged to vivado_build.log"
echo ""

cd $PROJ_DIR/vivado
vivado -mode batch -source build_all.tcl > $PROJ_DIR/vivado_build.log 2>&1

# Check result
if [ -f "$PROJ_DIR/vivado/lenet5_demo.bit" ]; then
    echo "  Bitstream: OK ($PROJ_DIR/vivado/lenet5_demo.bit)"
else
    echo "  ERROR: Bitstream generation failed!"
    echo "  Check: $PROJ_DIR/vivado_build.log"
    exit 1
fi

if [ -f "$PROJ_DIR/vivado/lenet5_demo.xsa" ]; then
    echo "  XSA: OK ($PROJ_DIR/vivado/lenet5_demo.xsa)"
else
    echo "  WARNING: XSA export failed"
fi
echo ""

#---------------------------------------------------------------------
# Step 2: Check timing and utilization
#---------------------------------------------------------------------
echo "[2/3] Build Reports"
grep -A5 "Design Timing Summary" $PROJ_DIR/vivado/impl_timing.rpt 2>/dev/null | head -10
grep -A3 "Slice LUTs\|Block RAM\|DSP" $PROJ_DIR/vivado/impl_util.rpt 2>/dev/null | head -15
echo ""

#---------------------------------------------------------------------
# Step 3: Instructions for firmware build
#---------------------------------------------------------------------
echo "[3/3] Next Steps - Firmware Build"
echo ""
echo "  === Option A: Vitis IDE (Recommended) ==="
echo "  1. Open Vitis IDE: vitis &"
echo "  2. Create Platform from: $PROJ_DIR/vivado/lenet5_demo.xsa"
echo "  3. Create Application (standalone OS, ps7_cortexa9_0)"
echo "  4. Add source: $PROJ_DIR/sw/src/main_ps_pl.c"
echo "  5. Build -> lenet5_demo.elf"
echo ""
echo "  === Option B: CLI (if Vitis Python API available) ==="
echo "  vitis -s $PROJ_DIR/scripts/build_vitis.py"
echo ""
echo "  === Deploy to SD Card ==="
echo "  1. Copy to FAT32 SD card:"
echo "     - lenet5_demo.bit (rename to system.bit or put in BOOT.bin)"
echo "     - lenet5_demo.elf"
echo "  2. Set J13 jumper: left two pins (SD Card boot)"
echo "  3. Connect UART (115200 baud, MIO48/49)"
echo "  4. Power on"
echo "  5. You'll see: '=== LeNet-5 Zynq PS+PL ===' on serial terminal"
echo ""
echo "============================================"
echo " Vivado build complete!"
echo "============================================"
