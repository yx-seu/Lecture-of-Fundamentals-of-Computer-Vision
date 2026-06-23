# LeNet-5 MNIST on Alinx AX7010

Real-time MNIST handwritten digit recognition on Alinx AX7010 (Zynq-7010, xc7z010clg400-1) using HLS-generated hardware accelerator.

To see the result, please open [demo report](./DEMO_REPORT.md).

## Architecture

```
PS (Cortex-A9)                   PL (FPGA)
+--------------------+    GP0    +-------------------+
| main.c             |<--------->| LeNet-5 HLS IP    |
| - image write      |  AXI4-Lite| - conv2d (5x5)    |
| - start/done poll  |           | - maxpool (2x2)   |
| - result read      |           | - fc (matvec)     |
| - UART output      |           | - ReLU            |
+--------------------+           +-------------------+

Register map (s_axi_control at 0x40000000):
  0x000: CTRL   (bit0=start, bit1=done, bit2=idle)
  0x010: out_scores[0..9]  (3 registers, packed int8)
  0x400: in_image[0..1023] (256 registers, 4 bytes per reg)
```

| Resource | Usage | Available | Util |
|----------|-------|-----------|------|
| LUT      | ~2800 | 17600     | 16%  |
| BRAM     | ~29   | 60        | 48%  |
| DSP      | ~42   | 80        | 52%  |
| Clock    | 100 MHz | -       | -    |
| Latency  | ~4.2 ms | -      | -    |

## Project Structure

```
├── hls/lenet5_accel/          # HLS C++ source
│   ├── src/                    #   conv2d, fc, pooling, types
│   ├── tb/                     #   testbench + test images
│   ├── hls_config.cfg
│   └── run.tcl
├── hls/weights/               # Quantized int8 weight headers
├── vivado/
│   ├── build_all.tcl           #   block design + synthesis + bitstream
│   └── alinx_ax7010.xdc        #   AX7010 pin constraints
├── sw/
│   ├── src/
│   │   ├── main.c              #   batch test firmware (100 images)
│   │   ├── test_batch.h        #   test data (100 MNIST images)
│   │   └── test_img_*.h        #   individual test images
│   └── lscript.ld              #   linker script
├── python/
│   ├── lenet5_model.py         #   LeNet-5 PyTorch model
│   ├── train_lenet5.py         #   training script
│   └── quantize_export.py      #   int8 quantization + C header export
├── scripts/
│   ├── setup_env.sh            #   environment setup
│   ├── run_demo_build.sh       #   end-to-end build
│   └── build_app.py            #   Vitis CLI firmware build
└── README.md
```

## Build Flow

### 0. Prerequisites

- Vivado/Vitis 2025.2
- Python 3.10+ with PyTorch, torchvision, numpy
- Alinx AX7010 development board

Source Vivado/Vitis environment before each step:
```bash
source <Vivado_install_dir>/settings64.sh
source <Vitis_install_dir>/settings64.sh
```

### 1. Train Model and Export Weights

```bash
cd python
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision numpy
python train_lenet5.py          # train LeNet-5 (>99% accuracy)
python quantize_export.py       # export int8 weights to ../hls/weights/
```

### 2. HLS Synthesis and IP Export

```bash
cd hls/lenet5_accel
vitis-run --tcl run.tcl --part xc7z010clg400-1
```

Output: `solution*/impl/ip/` (Vivado IP).

Note: Update paths in `run.tcl` and `hls_config.cfg` to match your environment.

### 3. Vivado Block Design, Synthesis, Bitstream

Update `vivado/build_all.tcl` with the correct path to the HLS IP directory, then:

```bash
cd vivado
vivado -mode batch -source build_all.tcl
```

Output: `lenet5_demo.bit`, `lenet5_demo.xsa`.

### 4. Build Firmware (Vitis IDE)

- Open Vitis IDE, create platform from `lenet5_demo.xsa`
- Create application component (standalone OS, ps7_cortexa9_0 processor)
- Add `sw/src/main.c` and `sw/src/test_batch.h` to the project
- Build; output: `lenet5_demo.elf`

### 5. Create BOOT.bin

Create a BIF file (`boot.bif`):
```
the_ROM_image:
{
    [bootloader]<platform_dir>/zynq_fsbl/fsbl.elf
    lenet5_demo.bit
    lenet5_demo.elf
}
```

```bash
bootgen -image boot.bif -arch zynq -process_bitstream bin -w -o BOOT.bin
```

*The BOOT.bin is all ready under repo for use.*

### 6. Deploy

Copy `BOOT.bin` to a FAT32-formatted SD card. Insert into AX7010. Set J13 jumper to the left two pins (SD boot). Connect UART (115200 baud, 8N1). Power on.

Expected output:
```
===== Batch Test (100 images) =====
Accuracy: 85/100 = 85%
  0: 8/8 = 100%
  1: 13/14 = 92%
  ...
```

## Board Setup

| Setting | Value |
|---------|-------|
| Board   | Alinx AX7010 (xc7z010clg400-1) |
| Boot    | J13 left two pins (SD card) |
| UART    | MIO 48 (TX) / MIO 49 (RX), 115200 8N1 |
| SD Card | FAT32, BOOT.bin in root |
| Power   | DC 5V |

## Key Design Decisions

- **Pure s_axilite** interface (no m_axi/DDR): avoids Zynq cache coherency and address mapping issues between CPU and HP port views
- **Resource-constrained** HLS (no PIPELINE/UNROLL): fits xc7z010 at 16% LUT with room for video pipeline
- **in_image at 0x400, out_scores at 0x10**: confirmed from HLS-generated hardware register map header
- **int8 quantization**: Q5.3 activations, Q0.7 weights; verified bit-accurate by C/RTL co-simulation (3/3 PASS)

## Demo

*See [demo report](./DEMO_REPORT.md) for more information.*

## References

- Alinx AX7010 User Manual: pin mappings and board specifications
- Xilinx UG902: Vivado HLS User Guide
- Xilinx UG585: Zynq-7000 Technical Reference Manual
