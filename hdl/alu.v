// Copyright (C) 2025 Miguel Marina
// Author: Miguel Marina <karel.capek.robotics@gmail.com>
// LinkedIn: https://www.linkedin.com/in/progman32/
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
