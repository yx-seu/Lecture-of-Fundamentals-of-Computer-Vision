# Vitis HLS Tcl Script for LeNet-5 Accelerator
open_project lenet5_accel
set_top lenet5_accel

# Source files (with include path for weights)
add_files src/types.h
add_files src/conv2d.h
add_files src/pooling.h
add_files src/fc.h
add_files src/weights.h
add_files src/lenet5_accel.cpp

# Testbench files
add_files -tb tb/lenet5_accel_tb.cpp
add_files -tb tb/test_image_0.h
add_files -tb tb/test_image_3.h
add_files -tb tb/test_image_7.h

# Solution
open_solution "solution15" -flow_target vivado
set_part {xc7z010clg400-1}
create_clock -period 10 -name default

# C Simulation
puts "=== C Simulation ==="
csim_design -clean

# C Synthesis
puts "=== C Synthesis ==="
csynth_design

# C/RTL Co-simulation (skip if synthesis fails)
puts "=== C/RTL Co-simulation ==="
cosim_design -trace_level all

# Export
puts "=== Export IP ==="
export_design -format ip_catalog

exit
