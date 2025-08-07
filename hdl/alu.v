`timescale 1ns/1ps
// Simple ALU supporting a small instruction set for the SoC CPU.
module alu(
    input  wire [3:0]  op,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] result,
    output wire        zero,
    output wire        gt
);
    // Operation codes
    localparam OP_ADD = 4'd0;
    localparam OP_SUB = 4'd1;
    localparam OP_AND = 4'd2;
    localparam OP_OR  = 4'd3;
    localparam OP_NOP = 4'd15;

    always @(*) begin
        case (op)
            OP_ADD: result = a + b;
            OP_SUB: result = a - b;
            OP_AND: result = a & b;
            OP_OR : result = a | b;
            default: result = 32'b0;
        endcase
    end

    assign zero = (result == 0);
    assign gt   = ($signed(a) > $signed(b));
endmodule
