`timescale 1ns/1ps
// Minimal PCIe bridge placeholder mirroring the Python PCIeBridge class.
module pcie_bridge(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] wdata,
    input  wire        we,
    output wire [31:0] rdata
);
    reg [31:0] reg0;
    always @(posedge clk) begin
        if (rst)
            reg0 <= 32'b0;
        else if (we)
            reg0 <= wdata;
    end
    assign rdata = reg0;
endmodule
