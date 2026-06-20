#=============================================================================
# Alinx AX7010 Pin Constraints (xc7z010clg400-1)
# Source: AX7010 User Manual
#=============================================================================

#--------------------------------------------------------------------
# PS System Clock (33.333 MHz, on-board oscillator) - PS_CLK_500
#--------------------------------------------------------------------
set_property PACKAGE_PIN E7  [get_ports ps_clk_33m]
set_property IOSTANDARD LVCMOS33 [get_ports ps_clk_33m]

#--------------------------------------------------------------------
# PL System Clock (50 MHz, on-board oscillator) - PL_GCLK
#--------------------------------------------------------------------
set_property PACKAGE_PIN U18 [get_ports pl_clk_50m]
set_property IOSTANDARD LVCMOS33 [get_ports pl_clk_50m]
create_clock -period 20.000 -name pl_clk [get_ports pl_clk_50m]

#--------------------------------------------------------------------
# Reset (PL Key4 = R17, active-low)
#--------------------------------------------------------------------
set_property PACKAGE_PIN R17 [get_ports pl_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports pl_rst_n]

#--------------------------------------------------------------------
# HDMI Output (BANK34, LVDS differential)
#--------------------------------------------------------------------
# Clock
set_property PACKAGE_PIN N18 [get_ports {hdmi_clk_p}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_clk_p}]
set_property PACKAGE_PIN P19 [get_ports {hdmi_clk_n}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_clk_n}]

# Data lane 0
set_property PACKAGE_PIN V20 [get_ports {hdmi_d0_p}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d0_p}]
set_property PACKAGE_PIN W20 [get_ports {hdmi_d0_n}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d0_n}]

# Data lane 1
set_property PACKAGE_PIN T20 [get_ports {hdmi_d1_p}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d1_p}]
set_property PACKAGE_PIN U20 [get_ports {hdmi_d1_n}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d1_n}]

# Data lane 2
set_property PACKAGE_PIN N20 [get_ports {hdmi_d2_p}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d2_p}]
set_property PACKAGE_PIN P20 [get_ports {hdmi_d2_n}]
set_property IOSTANDARD LVDS_33  [get_ports {hdmi_d2_n}]

# HDMI I2C (PL pins for ADV7511 EDID/control)
set_property PACKAGE_PIN R18 [get_ports hdmi_scl]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_scl]
set_property PACKAGE_PIN R16 [get_ports hdmi_sda]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_sda]

# HDMI CEC, HPD, Output Enable
set_property PACKAGE_PIN Y18 [get_ports hdmi_cec]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_cec]
set_property PACKAGE_PIN Y19 [get_ports hdmi_hpd]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_hpd]
set_property PACKAGE_PIN V16 [get_ports hdmi_out_en]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_out_en]

#--------------------------------------------------------------------
# OV5640 Camera (J10 Expansion Port, BANK34)
# DVP 8-bit data + HSYNC + VSYNC + PCLK
# Standard ALINX OV5640 module pinout on J10
#--------------------------------------------------------------------
# Camera pixel clock (input)
set_property PACKAGE_PIN P14 [get_ports cam_pclk]
set_property IOSTANDARD LVCMOS33 [get_ports cam_pclk]

# Camera HSYNC
set_property PACKAGE_PIN N17 [get_ports cam_href]
set_property IOSTANDARD LVCMOS33 [get_ports cam_href]

# Camera VSYNC
set_property PACKAGE_PIN P18 [get_ports cam_vsync]
set_property IOSTANDARD LVCMOS33 [get_ports cam_vsync]

# Camera DVP data [7:0] (J10 pins mapped to BANK34)
set_property PACKAGE_PIN Y17 [get_ports {cam_data[0]}]
set_property PACKAGE_PIN Y16 [get_ports {cam_data[1]}]
set_property PACKAGE_PIN W15 [get_ports {cam_data[2]}]
set_property PACKAGE_PIN V15 [get_ports {cam_data[3]}]
set_property PACKAGE_PIN Y14 [get_ports {cam_data[4]}]
set_property PACKAGE_PIN W14 [get_ports {cam_data[5]}]
set_property PACKAGE_PIN T15 [get_ports {cam_data[6]}]
set_property PACKAGE_PIN T14 [get_ports {cam_data[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {cam_data[*]}]

# Camera I2C (use PL GPIO for bit-bang or PS I2C via EMIO)
set_property PACKAGE_PIN U15 [get_ports cam_scl]
set_property IOSTANDARD LVCMOS33 [get_ports cam_scl]
set_property PACKAGE_PIN U14 [get_ports cam_sda]
set_property IOSTANDARD LVCMOS33 [get_ports cam_sda]

# Camera reset / power-down
set_property PACKAGE_PIN P16 [get_ports cam_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports cam_rst_n]
set_property PACKAGE_PIN P15 [get_ports cam_pwdn]
set_property IOSTANDARD LVCMOS33 [get_ports cam_pwdn]

# Camera XCLK (output, typically 24MHz for OV5640)
set_property PACKAGE_PIN W18 [get_ports cam_xclk]
set_property IOSTANDARD LVCMOS33 [get_ports cam_xclk]

#--------------------------------------------------------------------
# PL LEDs (BANK35, active-low)
#--------------------------------------------------------------------
set_property PACKAGE_PIN M14 [get_ports pl_led[0]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_led[0]]
set_property PACKAGE_PIN M15 [get_ports pl_led[1]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_led[1]]
set_property PACKAGE_PIN K16 [get_ports pl_led[2]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_led[2]]
set_property PACKAGE_PIN J16 [get_ports pl_led[3]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_led[3]]

#--------------------------------------------------------------------
# PL Keys (BANK34/35, active-low)
#--------------------------------------------------------------------
set_property PACKAGE_PIN N15 [get_ports pl_key[0]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_key[0]]
set_property PACKAGE_PIN N16 [get_ports pl_key[1]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_key[1]]
set_property PACKAGE_PIN T17 [get_ports pl_key[2]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_key[2]]
set_property PACKAGE_PIN R17 [get_ports pl_key[3]]
set_property IOSTANDARD LVCMOS33 [get_ports pl_key[3]]

#--------------------------------------------------------------------
# PS MIO Configuration (via Zynq7 PS IP in block design)
# These are handled automatically by the PS7 configuration:
#   UART1:     MIO48(TX=B12), MIO49(RX=C12)
#   I2C0:      MIO10(SCL=E9), MIO11(SDA=C6)
#   SD0:       MIO40(D14)~MIO45(B15), MIO47(B14)=CD
#   PS LED:    MIO0(E6), MIO13(E8) — GPIO outputs
#   PS Keys:   MIO50(B13), MIO51(B9) — GPIO inputs
#   QSPI:      MIO1(A7)~MIO6(A5)
#   DDR3:      BANK502 (fixed, auto-routed)
#   PS_CLK:    E7 (33.333MHz, auto-routed)
#--------------------------------------------------------------------
