`timescale 1ns/1ps
// Wishbone-like memory module mirroring the Python WishboneMemory class.
module wishbone_memory #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire [ADDR_WIDTH-1:0]    addr,
    input  wire [DATA_WIDTH-1:0]    wdata,
    input  wire                     we,
    output reg  [DATA_WIDTH-1:0]    rdata
);
    // simple synchronous memory
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= wdata;
        end
        rdata <= mem[addr];
    end
endmodule
