#!/usr/bin/env python3
"""Full build: replace main.c, rebuild app, create boot image"""
import vitis, os, shutil, subprocess

WS   = "/home/aika/cv/sw/vitis_ws"
APP  = "lenet5_demo"
SRC  = "/home/aika/cv/sw/vitis_ws/lenet5_demo/src/main.c"
DIAG = "/home/aika/cv/sw/src/main_pltest.c"

c = vitis.create_client()
c.set_workspace(WS)

# 1. Replace main.c
shutil.copy(DIAG, SRC)
print(f"Replaced: {SRC}")

# 2. Get app and build
app = c.get_component(name=APP)
print(f"Building {APP}...")
app.build()
print("Build complete")

# 3. Find ELF
elf = None
for root, dirs, files in os.walk(os.path.join(WS, APP)):
    for f in files:
        if f.endswith('.elf') and 'fsbl' not in root:
            elf = os.path.join(root, f)
            break
    if elf: break

if not elf:
    print("ELF not found!")
    c.close()
    exit(1)

print(f"ELF: {elf}")
shutil.copy(elf, "/home/aika/cv/sw/lenet5_demo.elf")

# 4. Create BOOT.BIN
bit = "/home/aika/cv/lenet5_demo.bit"
fsbl = None
for root, dirs, files in os.walk(WS):
    for f in files:
        if f == "fsbl.elf":
            fsbl = os.path.join(root, f)
            break
    if fsbl: break

if fsbl:
    # Write bif file
    bif = "/tmp/boot.bif"
    with open(bif, "w") as f:
        f.write(f"""// BOOT.BIN for LeNet-5 demo
the_ROM_image:
{{
    [bootloader]{fsbl}
    {bit}
    {elf}
}}
""")
    print(f"BIF written: {bif}")

    # Create boot image
    out_dir = "/mnt/c/Users/17156/Desktop/lenet5"
    os.makedirs(out_dir, exist_ok=True)
    boot_bin = os.path.join(out_dir, "BOOT.bin")

    # Use bootgen from Vitis
    bootgen = "/home/aika/AMD/2025.2/Vitis/bin/bootgen"
    result = subprocess.run([bootgen, "-image", bif, "-arch", "zynq",
                              "-process_bitstream", "bin",
                              "-w", "-o", boot_bin],
                             capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print(f"\nBOOT.BIN created: {boot_bin}")
    else:
        print(f"bootgen error: {result.stderr}")
else:
    print("FSBL not found, skipping BOOT.BIN")

c.close()
print("\nDONE")
