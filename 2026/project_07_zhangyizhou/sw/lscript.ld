/*
 * Minimal linker script for LeNet-5 bare-metal application
 * Zynq-7010: DDR 512MB at 0x00100000
 */
MEMORY {
    ps7_ddr_0 : ORIGIN = 0x00100000, LENGTH = 0x1FF00000
    ps7_ram_0 : ORIGIN = 0x00000000, LENGTH = 0x00030000
}

SECTIONS {
    .vectors : {
        KEEP(*(.vectors))
    } > ps7_ddr_0

    .text : {
        *(.text*)
        *(.gnu.linkonce.t.*)
    } > ps7_ddr_0

    .rodata : {
        *(.rodata*)
    } > ps7_ddr_0

    .data : {
        *(.data*)
    } > ps7_ddr_0

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > ps7_ddr_0

    .heap : {
        _heap_start = .;
        . = . + 0x100000;
        _heap_end = .;
    } > ps7_ddr_0

    .stack : {
        __stack_end = .;
        . = . + 0x10000;
        __stack = .;
    } > ps7_ddr_0

    _end = .;
}
ENTRY(_vector_table)
