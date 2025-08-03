`timescale 1ns/1ps
// Minimal pass-through MMU mirroring the Python MMU class.
module mmu(
    input  wire [31:0] va,
    input  wire        mem_write,
    output wire [31:0] pa,
    output wire        fault
);
    // In this simplified model the MMU performs identity translation
    // and never signals a page fault.
    assign pa    = va;
    assign fault = 1'b0;
endmodule
