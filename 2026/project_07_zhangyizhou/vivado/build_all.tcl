# Vivado: Zynq7 PS + LeNet-5 s_axilite only (no m_axi)
set part    xc7z010clg400-1
set proj    ./lenet5_build
set hls_ip  /home/aika/cv/hls/lenet5_accel/lenet5_accel/solution15/impl/ip

create_project lenet5_build $proj -part $part -force
set_property ip_repo_paths $hls_ip [current_project]
update_ip_catalog
create_bd_design "design_1"

# PS: GP0 only, no HP0 needed
set zynq [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_SD0_IO {MIO 40 .. 45} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_FPGA_FCLK0_ENABLE {1} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_QSPI_IO {MIO 1 .. 6} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {0} \
] $zynq

set lenet5 [create_bd_cell -type ip -vlnv xilinx.com:hls:lenet5_accel:1.0 lenet5_0]
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_0]

set clk [create_bd_net clk_100mhz]
connect_bd_net -net $clk [get_bd_pins ps7_0/FCLK_CLK0]
connect_bd_net -net $clk [get_bd_pins ps7_0/M_AXI_GP0_ACLK]
connect_bd_net -net $clk [get_bd_pins rst_0/slowest_sync_clk]
connect_bd_net -net $clk [get_bd_pins lenet5_0/ap_clk]
connect_bd_net -net $clk [get_bd_pins smc_0/aclk]

connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins lenet5_0/ap_rst_n]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins smc_0/aresetn]

connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] [get_bd_intf_pins smc_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_0/M00_AXI] [get_bd_intf_pins lenet5_0/s_axi_control]

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR"} $zynq
assign_bd_address; validate_bd_design; save_bd_design
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 1; wait_on_run synth_1; open_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 1; wait_on_run impl_1; open_run impl_1
set bf [glob -nocomplain [file join $proj ${proj}.runs impl_1 *.bit]]
if {$bf ne ""} { file copy -force $bf ./lenet5_demo.bit }
write_hw_platform -fixed -include_bit -force -file ./lenet5_demo.xsa
puts "=== BUILD COMPLETE ==="
